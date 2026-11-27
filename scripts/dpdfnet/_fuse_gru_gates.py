"""Fuse GRU-cell input and recurrent gate projections with exact bias semantics."""
from pathlib import Path
import json
import numpy as np
import onnx
from onnx import helper,numpy_helper

HERE=Path(__file__).resolve().parent
m=onnx.load(HERE/"core-batched.onnx")
by_name={n.name:n for n in m.graph.node}
arrays={a.name:numpy_helper.to_array(a) for a in m.graph.initializer}
replace={};removed=set();report=[]
for n in m.graph.node:
    if n.op_type!="Gemm" or not n.name.endswith("/grucell/Gemm"): continue
    stem=n.name.removesuffix("/Gemm")
    h=by_name[stem+"/Gemm_1"]
    a=arrays[n.input[1]];b=arrays[h.input[1]]
    ba=arrays[n.input[2]];bb=arrays[h.input[2]]
    hidden=b.shape[1];width=a.shape[1]
    assert a.shape[0]==b.shape[0]==3*hidden
    weights=np.zeros((4*hidden,width+hidden),np.float32)
    weights[:2*hidden,:width]=a[:2*hidden]
    weights[:2*hidden,width:]=b[:2*hidden]
    weights[2*hidden:3*hidden,:width]=a[2*hidden:]
    weights[3*hidden:,width:]=b[2*hidden:]
    bias=np.r_[ba[:2*hidden]+bb[:2*hidden],ba[2*hidden:],bb[2*hidden:]]
    prefix=stem+"/fused_gates"
    for name,arr in [
        (prefix+"_w",weights),(prefix+"_b",bias),
        (prefix+"_sections",np.full(4,hidden,np.int64))]:
        m.graph.initializer.append(numpy_helper.from_array(arr,name))
    split_x=by_name[stem+"/Split"]
    split_h=by_name[stem+"/Split_1"]
    add_r=by_name[stem+"/Add"];add_z=by_name[stem+"/Add_1"]
    assert set(add_r.input)=={split_x.output[0],split_h.output[0]}
    assert set(add_z.input)=={split_x.output[1],split_h.output[1]}
    # Insert only after both source projections' inputs have been produced.
    replace[h.name]=[
        helper.make_node("Concat",[n.input[0],h.input[0]],[prefix+"_input"],name=prefix+"_concat",axis=1),
        helper.make_node("Gemm",[prefix+"_input",prefix+"_w",prefix+"_b"],[prefix+"_projection"],name=prefix+"_gemm",transB=1),
        helper.make_node("Split",[prefix+"_projection",prefix+"_sections"],
             [add_r.output[0],add_z.output[0],split_x.output[2],split_h.output[2]],name=prefix+"_split",axis=1)]
    removed.update([n.name,split_x.name,split_h.name,add_r.name,add_z.name])
    report.append(stem)
nodes=[]
for n in m.graph.node:
    if n.name in removed: continue
    nodes.extend(replace[n.name] if n.name in replace else [n])
del m.graph.node[:];m.graph.node.extend(nodes)
needed={x.name for x in m.graph.output};live=[]
for n in reversed(m.graph.node):
    if any(o in needed for o in n.output):
        live.append(n);needed.update(n.input)
del m.graph.node[:];m.graph.node.extend(reversed(live))
initializers=[a for a in m.graph.initializer if a.name in needed]
del m.graph.initializer[:];m.graph.initializer.extend(initializers)
del m.graph.value_info[:]
onnx.checker.check_model(m)
onnx.save(m,HERE/"core-gates.onnx")

import onnxruntime as ort
opts=ort.SessionOptions();opts.intra_op_num_threads=1
ref=ort.InferenceSession(str(HERE/"core.onnx"),opts,providers=["CPUExecutionProvider"])
target=ort.InferenceSession(str(HERE/"core-gates.onnx"),opts,providers=["CPUExecutionProvider"])
pre=ort.InferenceSession(str(HERE/"preprocess.onnx"),opts,providers=["CPUExecutionProvider"])
a=np.fromfile(HERE/"initial-state.bin",np.float32);b=a.copy()
rng=np.random.default_rng(47);worst=np.zeros(2)
for k in range(100):
    x=rng.normal(0,.8,(1,1,161,2)).astype(np.float32)
    if k<4:x*=0
    ef,sf,es,ss=pre.run(None,{"spec":x,"state_in":a})
    a[:128]=b[:128]=np.r_[es,ss]
    inputs={"spec":x,"state_in":a,"erb_feat":ef,"spec_feat":sf}
    ya,a=ref.run(None,inputs)
    inputs["state_in"]=b
    yb,b=target.run(None,inputs)
    worst=np.maximum(worst,[np.max(abs(ya-yb)),np.max(abs(a-b))])
assert worst.max()<2e-4,worst
(HERE/"gate-validation.json").write_text(json.dumps({"gate_groups":report,"frames":100,"max_absolute_error":worst.tolist()},indent=2)+"\n")
print("GRU gates fused",len(report),"max errors",worst,flush=True)
