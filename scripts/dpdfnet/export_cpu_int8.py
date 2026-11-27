"""Export DPDFNet2 with only five large GRU input projections quantized to INT8."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import onnx
from onnx import helper, numpy_helper
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic

SOURCE_SHA256 = "614ae61a58e948b594ca52d48e7712fb0435a81923a0a50ba1b7a463034eb617"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if hashlib.sha256(args.source.read_bytes()).hexdigest() != SOURCE_SHA256:
        parser.error("Expected the validated DPDFNet2 FP32 CPU model.")
    args.output.mkdir(parents=True, exist_ok=True)
    model = onnx.load(args.source)
    arrays = {a.name: numpy_helper.to_array(a) for a in model.graph.initializer}
    nodes, selected = [], []
    for node in model.graph.node:
        if node.op_type != "Gemm":
            nodes.append(node)
            continue
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        assert attrs.get("transA", 0) == 0
        assert attrs.get("alpha", 1) == attrs.get("beta", 1) == 1
        weights = arrays[node.input[1]]
        weights = weights.T if attrs.get("transB", 0) else weights
        name = node.name + "/int8_weight_source"
        model.graph.initializer.append(numpy_helper.from_array(weights.copy(), name))
        temp = node.name + "/matmul_output"
        nodes.extend([
            helper.make_node("MatMul", [node.input[0], name], [temp], name=node.name),
            helper.make_node("Add", [temp, node.input[2]], list(node.output),
                             name=node.name + "/bias"),
        ])
        if weights.shape[0] >= 256 and weights.shape[1] >= 768 and not node.name.endswith("_1"):
            selected.append(node.name)
    assert len(selected) == 5
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    needed = {name for node in nodes for name in node.input}
    live = [a for a in model.graph.initializer if a.name in needed]
    del model.graph.initializer[:]
    model.graph.initializer.extend(live)
    onnx.checker.check_model(model)
    target = args.output / "dpdfnet2-cpu-int8.onnx"
    with tempfile.TemporaryDirectory(prefix="dpdfnet-int8-") as temp:
        rewritten = Path(temp) / "fp32.onnx"
        onnx.save(model, rewritten)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        ref = ort.InferenceSession(str(args.source), options, providers=["CPUExecutionProvider"])
        test = ort.InferenceSession(str(rewritten), options, providers=["CPUExecutionProvider"])
        meta = {p.key: p.value for p in model.metadata_props}
        a = np.zeros(45424, np.float32)
        a[:32] = np.fromstring(meta["erb_norm_init"], sep=",", dtype=np.float32)
        a[32:128] = np.fromstring(meta["spec_norm_init"], sep=",", dtype=np.float32)
        b = a.copy()
        rng = np.random.default_rng(47)
        worst = np.zeros(2)
        for frame in range(40):
            x = rng.normal(0, .8, (1, 1, 161, 2)).astype(np.float32)
            if frame < 3:
                x *= 0
            y, a = ref.run(None, {"spec": x, "state_in": a})
            z, b = test.run(None, {"spec": x, "state_in": b})
            worst = np.maximum(worst, [np.max(abs(y-z)), np.max(abs(a-b))])
        assert worst.max() < 2e-4, worst
        quantize_dynamic(str(rewritten), str(target),
                         op_types_to_quantize=["MatMul"],
                         nodes_to_quantize=selected, per_channel=True,
                         weight_type=QuantType.QInt8,
                         extra_options={"MatMulConstBOnly": True})
    quantized = onnx.load(target)
    assert sum(n.op_type == "MatMulInteger" for n in quantized.graph.node) == 5
    metadata = {
        "source_cpu_sha256": SOURCE_SHA256,
        "method": "dynamic U8S8, per-channel weights",
        "quantized_input_projections": selected,
        "recurrent_state_projections": "FP32",
        "feature_normalization": "FP32",
        "fp32_rewrite_check": {"frames": 40, "max_absolute_error": worst.tolist()},
    }
    manifest_path = args.output / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest["cpu_int8"] = metadata
    manifest.setdefault("files", {})[target.name] = hashlib.sha256(target.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output / "SHA256SUMS").write_text(
        "".join(f"{sha}  {name}\n" for name, sha in manifest["files"].items())
    )
    print(json.dumps(metadata, indent=2))
    print(target)


if __name__ == "__main__":
    main()
