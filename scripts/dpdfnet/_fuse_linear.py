"""Fuse independent grouped linear layers into one batched matrix operation."""
from pathlib import Path
import json
import sys
import numpy as np
import onnx
from onnx import helper,numpy_helper

HERE=Path(__file__).resolve().parent
cpu_mode = len(sys.argv) > 1 and sys.argv[1] == "--cpu"
source_name = "original.onnx" if cpu_mode else "core.onnx"
output_name = "original-batched.onnx" if cpu_mode else "core-batched.onnx"
m=onnx.load(HERE/source_name)
producer={o:n for n in m.graph.node for o in n.output}
values={a.name:numpy_helper.to_array(a) for a in m.graph.initializer}
removed=set();replaced={};report=[]
for n in m.graph.node:
    if n.op_type!="Concat" or len(n.input)<8: continue
    group=[producer.get(x) for x in n.input]
    if not all(g is not None and g.op_type=="Gemm" for g in group): continue
    splits=[producer.get(g.input[0]) for g in group]
    if not all(s is splits[0] for s in splits) or splits[0].op_type!="Split": continue
    split=splits[0]
    assert list(split.output)==[g.input[0] for g in group]
    weights=[];biases=[]
    for g in group:
        attrs={a.name:helper.get_attribute_value(a) for a in g.attribute}
        assert attrs.get("transB",0)==1 and attrs.get("transA",0)==0
        assert attrs.get("alpha",1)==attrs.get("beta",1)==1
        weights.append(values[g.input[1]])
        biases.append(values[g.input[2]] if len(g.input)>2 else np.zeros(weights[-1].shape[0],np.float32))
    row=sum(w.shape[0] for w in weights)
    col=sum(w.shape[1] for w in weights)
    prefix=n.name+"/fused"
    wn,bn=prefix+"_weight",prefix+"_bias"
    assert all(w.shape==weights[0].shape for w in weights)
    channels_out,channels_in=weights[0].shape
    m.graph.initializer.extend([
        numpy_helper.from_array(np.stack([w.T for w in weights]),wn),
        numpy_helper.from_array(np.stack(biases)[:,None,:],bn),
        numpy_helper.from_array(np.array([len(group),1,channels_in],np.int64),prefix+"_in_shape"),
        numpy_helper.from_array(np.array([1,row],np.int64),prefix+"_out_shape")])
    replaced[n.name]=[
        helper.make_node("Reshape",[split.input[0],prefix+"_in_shape"],[prefix+"_input"],name=prefix+"_reshape_in"),
        helper.make_node("MatMul",[prefix+"_input",wn],[prefix+"_product"],name=prefix+"_matmul"),
        helper.make_node("Add",[prefix+"_product",bn],[prefix+"_biased"],name=prefix+"_bias"),
        helper.make_node("Reshape",[prefix+"_biased",prefix+"_out_shape"],list(n.output),name=prefix+"_reshape_out")]
    removed.update([g.name for g in group]+[split.name])
    report.append({"name":n.name,"groups":len(group),"input_features":col,"output_features":row})
nodes=[]
for n in m.graph.node:
    if n.name in removed: continue
    nodes.extend(replaced[n.name] if n.name in replaced else [n])
del m.graph.node[:]
m.graph.node.extend(nodes)
# Keep only tensors that lead to public outputs.
needed={x.name for x in m.graph.output}
live=[]
for n in reversed(m.graph.node):
    if any(o in needed for o in n.output):
        live.append(n)
        needed.update(n.input)
del m.graph.node[:]
m.graph.node.extend(reversed(live))
keep=[a for a in m.graph.initializer if a.name in needed]
del m.graph.initializer[:]
m.graph.initializer.extend(keep)
del m.graph.value_info[:]
onnx.checker.check_model(m)
onnx.save(m,HERE/output_name)

import onnxruntime as ort

if cpu_mode:
    opts=ort.SessionOptions()
    opts.intra_op_num_threads=1
    ref=ort.InferenceSession(str(HERE/source_name),opts,providers=["CPUExecutionProvider"])
    target=ort.InferenceSession(str(HERE/output_name),opts,providers=["CPUExecutionProvider"])
    a=np.fromfile(HERE/"initial-state.bin",np.float32)
    b=a.copy()
    rng=np.random.default_rng(47)
    worst=np.zeros(2)
    for k in range(100):
        x=rng.normal(0,.8,(1,1,161,2)).astype(np.float32)
        if k<4:
            x*=0
        ya,a=ref.run(None,{"spec":x,"state_in":a})
        yb,b=target.run(None,{"spec":x,"state_in":b})
        worst=np.maximum(worst,[np.max(abs(ya-yb)),np.max(abs(a-b))])
    assert worst.max()<2e-4,worst
    (HERE/"cpu-validation.json").write_text(json.dumps({"frames":100,"max_absolute_error":worst.tolist()},indent=2)+"\n")
    print("CPU batched projection validated",worst,flush=True)
    sys.exit(0)
opts=ort.SessionOptions();opts.intra_op_num_threads=1
a=ort.InferenceSession(str(HERE/"core.onnx"),opts,providers=["CPUExecutionProvider"])
b=ort.InferenceSession(str(HERE/"core-batched.onnx"),opts,providers=["CPUExecutionProvider"])
pre=ort.InferenceSession(str(HERE/"preprocess.onnx"),opts,providers=["CPUExecutionProvider"])
state=np.fromfile(HERE/"initial-state.bin",np.float32)
state_a=state.copy();state_b=state.copy()
rng=np.random.default_rng(47);worst=np.zeros(2)
for i in range(100):
    x=rng.normal(0,.8,(1,1,161,2)).astype(np.float32)
    if i<4: x*=0
    features=pre.run(None,{"spec":x,"state_in":state_a})
    ef,sf,es,ss=features
    state_a[:128]=state_b[:128]=np.r_[es,ss]
    inputs={"spec":x,"state_in":state_a,"erb_feat":ef,"spec_feat":sf}
    ya,state_a=a.run(None,inputs)
    inputs["state_in"]=state_b
    yb,state_b=b.run(None,inputs)
    worst=np.maximum(worst,[np.max(abs(ya-yb)),np.max(abs(state_a-state_b))])
assert worst.max()<2e-4,worst
(HERE/"batched-validation.json").write_text(json.dumps({"fusions":report,"frames":100,"max_absolute_error":worst.tolist()},indent=2)+"\n")
print("Fused",len(report),"groups",sum(x["groups"] for x in report),"max error",worst,flush=True)
