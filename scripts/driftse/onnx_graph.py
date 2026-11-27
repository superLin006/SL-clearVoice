"""DriftSE graph transforms for streaming inference."""

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def separate_states(model):
    graph = onnx.shape_inference.infer_shapes(model).graph
    by_value = {output: node for node in graph.node for output in node.output}
    packed = by_value["next_state"]
    [split] = [node for node in graph.node if "state" in node.input]
    assert packed.op_type == "Concat" and split.op_type == "Split"
    assert len(packed.input) == len(split.output)
    assert [item.name for item in graph.input] == ["spectrum", "state"]
    assert [item.name for item in graph.output] == ["estimate", "next_state"]
    del graph.input[1:]
    del graph.output[1:]
    replacements = {}
    omitted = {split.name, packed.name}
    for index, (part, next_part) in enumerate(zip(split.output, packed.input)):
        [restore] = [node for node in graph.node if part in node.input]
        assert restore.op_type == "Reshape", restore.name
        pack = by_value[next_part]
        assert pack.op_type == "Reshape", pack.name
        info = by_value[restore.input[1]].attribute[0].t
        shape = numpy_helper.to_array(info).tolist()
        name = f"state_{index:02d}"
        replacements[restore.output[0]] = name
        omitted.update((restore.name, pack.name))
        graph.input.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, shape))
        graph.output.append(helper.make_tensor_value_info(pack.input[0], TensorProto.FLOAT, shape))

    # A state's output feeds the corresponding input on the next 8-frame call.
    retained = [node for node in graph.node if node.name not in omitted]
    live = {item.name for item in graph.output}
    used_nodes = []
    for node in reversed(retained):
        if live.intersection(node.output):
            for index, name in enumerate(node.input):
                node.input[index] = replacements.get(name, name)
            live.update(node.input)
            used_nodes.append(node)
    assert all(item.name in live for item in graph.input)
    del graph.node[:]
    graph.node.extend(reversed(used_nodes))
    names = {value for node in graph.node for value in node.output}
    infos = [item for item in graph.value_info if item.name in names]
    del graph.value_info[:]
    graph.value_info.extend(infos)
    model.graph.CopyFrom(graph)


def rewrite_projection(model):
    graph = model.graph
    nodes = {n.name: n for n in graph.node}
    weights = {w.name: w for w in graph.initializer}
    users = {}
    for node in graph.node:
        for value in node.input:
            users.setdefault(value, []).append(node.name)
    patches, removals = {}, set()

    frequency_nodes = [node for node in graph.node
                       if node.op_type == "LSTM" and "/intra_rnn" in node.name]
    assert len(frequency_nodes) == 4
    for block, recurrent in enumerate(frequency_nodes):
        prefix = f"/frequency_block.{block}"
        chain = []
        last = recurrent
        for op_type in ("Transpose", "Reshape", "Transpose", "ConvTranspose", "Reshape", "Transpose"):
            [user] = users[last.output[0]]
            last = nodes[user]
            assert last.op_type == op_type, last.name
            chain.append(last)
        parents = [recurrent, *chain[:-1]]
        for parent, child in zip(parents, chain):
            assert child.input[0] == parent.output[0]
            assert users[parent.output[0]] == [child.name], parent.name
        assert list(chain[0].attribute[0].ints) == [0, 2, 1, 3]
        assert list(chain[2].attribute[0].ints) == [1, 2, 0]
        assert list(chain[5].attribute[0].ints) == [0, 1, 3, 2]
        conv_weights = numpy_helper.to_array(weights[chain[3].input[1]])
        assert conv_weights.shape == (200, 32, 4)
        conv_attrs = {attr.name: helper.get_attribute_value(attr) for attr in chain[3].attribute}
        assert conv_attrs == {"dilations": [1], "group": 1, "kernel_shape": [4],
                              "pads": [0, 0], "strides": [1]}
        conv_bias = numpy_helper.to_array(weights[chain[3].input[2]])
        assert conv_bias.shape == (32,)

        branch = []

        def constant(name, values):
            tensor_name = f"{prefix}/sequence_projection/{name}"
            graph.initializer.append(numpy_helper.from_array(np.asarray(values), name=tensor_name))
            return tensor_name

        def op(kind, name, *inputs, output=None, **attrs):
            name = f"{prefix}/sequence_projection/{name}"
            result = output or name
            branch.append(helper.make_node(kind, list(inputs), [result], name=name, **attrs))
            return result

        directions = []
        for direction in range(2):
            hidden = op("Gather", f"extract-direction-{direction}", recurrent.output[0],
                        constant(f"direction-{direction}", np.int64(direction)), axis=1)
            start = direction * 100
            # ConvTranspose1d with stride one is four per-bin linear
            # projections added at frequency offsets 0..3.
            taps = conv_weights[start:start + 100].transpose(0, 2, 1).copy().reshape(100, 128)
            matrix = constant(f"taps-{direction}", taps)
            directions.append(op("MatMul", f"project-direction-{direction}", hidden, matrix))
        projected = op("Add", "sum-directions", *directions)
        chunks = []
        for tap in range(4):
            start, end = 32 * tap, 32 * (tap + 1)
            values = op("Slice", f"tap-{tap}", projected, constant(f"start-{tap}", [start]),
                        constant(f"stop-{tap}", [end]), constant(f"axis-{tap}", [2]))
            pads = constant(f"pads-{tap}", [tap, 0, 0, 3 - tap, 0, 0])
            chunks.append(op("Pad", f"offset-{tap}", values, pads, mode="constant"))
        first_pair = op("Add", "first-pair", *chunks[:2])
        second_pair = op("Add", "second-pair", *chunks[2:])
        combined = op("Add", "sum-taps", first_pair, second_pair)
        shifted = op("Add", "bias", combined, constant("bias-value", conv_bias.reshape(1, 1, 32)))
        smaller = op("Transpose", "restore-time-major", shifted, perm=[1, 0, 2])
        op("Unsqueeze", "batch-axis", smaller, constant("unsqueeze-axis", [0]),
           output=chain[-1].output[0])
        removals.update(node.name for node in chain)
        patches[chain[-1].name] = branch

    original = list(graph.node)
    del graph.node[:]
    for node in original:
        if node.name not in removals:
            graph.node.append(node)
        graph.node.extend(patches.get(node.name, []))
    # Delete dead nodes and initializers so the compiler sees just the new path.
    relevant = {value.name for value in graph.output}
    retained = []
    for node in reversed(graph.node):
        if relevant.intersection(node.output):
            retained.append(node)
            relevant.update(value for value in node.input if value)
    del graph.node[:]
    graph.node.extend(reversed(retained))
    initializers = [w for w in graph.initializer if w.name in relevant]
    del graph.initializer[:]
    graph.initializer.extend(initializers)
    del graph.value_info[:]
