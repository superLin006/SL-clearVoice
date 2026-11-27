"""Separate DPDFNet's precision-sensitive feature normalization from the NPU core."""
from pathlib import Path
import copy,json
import numpy as np
import onnx
from onnx import helper,numpy_helper,shape_inference,utils

HERE=Path(__file__).resolve().parent
model=shape_inference.infer_shapes(onnx.load(HERE/"original.onnx"))
meta={p.key:p.value for p in model.metadata_props}
state=np.zeros(int(meta["state_size"]),np.float32)
state[:32]=np.fromstring(meta["erb_norm_init"],sep=",",dtype=np.float32)
state[32:128]=np.fromstring(meta["spec_norm_init"],sep=",",dtype=np.float32)
state.tofile(HERE/"initial-state.bin")
erb="/model/erb_norm/Div_output_0"
spec="/model/spec_norm/Concat_output_0"
norms=["/model/erb_norm/Reshape_1_output_0","/model/spec_norm/Reshape_1_output_0"]
onnx.save(model,HERE/"inferred.onnx")
utils.extract_model(str(HERE/"inferred.onnx"),str(HERE/"preprocess.onnx"),
                    ["spec","state_in"],[erb,spec,*norms])
core=copy.deepcopy(model)
mapping={norms[0]:"/model/erb_norm/Slice_output_0",norms[1]:"/model/spec_norm/Slice_output_0"}
for node in core.graph.node:
    for k,value in enumerate(node.input):
        if value in mapping:
            node.input[k]=mapping[value]
onnx.save(core,HERE/"core-pre-extract.onnx")
utils.extract_model(str(HERE/"core-pre-extract.onnx"),str(HERE/"core.onnx"),
                    ["spec","state_in",erb,spec],["spec_e","state_out"])
core=onnx.load(HERE/"core.onnx")
names={"spec":"spec","state_in":"state_in",erb:"erb_feat",spec:"spec_feat"}
for node in core.graph.node:
    for tensor in (node.input,node.output):
        for k,v in enumerate(tensor):
            if v in names: tensor[k]=names[v]
for tensor in list(core.graph.input)+list(core.graph.output)+list(core.graph.value_info):
    if tensor.name in names: tensor.name=names[tensor.name]
onnx.checker.check_model(core)
onnx.save(core,HERE/"core.onnx")
fb=next(x for x in model.graph.initializer if x.name=="model.erb_fb")
numpy_helper.to_array(fb).astype(np.float32).tofile(HERE/"erb-filterbank.bin")
print("Core exported",[(x.name,[d.dim_value for d in x.type.tensor_type.shape.dim]) for x in core.graph.input],len(core.graph.node))

import onnxruntime as ort
opts=ort.SessionOptions()
opts.intra_op_num_threads=1
ref=ort.InferenceSession(str(HERE/"original.onnx"),opts,providers=["CPUExecutionProvider"])
pre=ort.InferenceSession(str(HERE/"preprocess.onnx"),opts,providers=["CPUExecutionProvider"])
target=ort.InferenceSession(str(HERE/"core.onnx"),opts,providers=["CPUExecutionProvider"])
rng=np.random.default_rng(47)
a=state.copy();b=state.copy();errors=[]
examples=[]
for k in range(80):
    x=rng.normal(0,.8,(1,1,161,2)).astype(np.float32)
    if k<3: x*=0
    y,a=ref.run(None,{"spec":x,"state_in":a})
    ef,sf,es,ss=pre.run(None,{"spec":x,"state_in":b})
    b[:128]=np.r_[es,ss]
    feeds={"spec":x,"state_in":b,"erb_feat":ef,"spec_feat":sf}
    if k in (0,3,12):
        case=HERE/f"case-{k}"
        case.mkdir(exist_ok=True)
        for i,name in enumerate(("spec","state_in","erb_feat","spec_feat")):
            feeds[name].tofile(case/f"input-{i}.bin")
        y.tofile(case/"reference-0.bin")
        a.tofile(case/"reference-1.bin")
    z,b=target.run(None,feeds)
    errors.append([float(np.max(abs(z-y))),float(np.max(abs(b-a)))])
assert np.max(errors)<1e-5, errors
(HERE/"export-validation.json").write_text(json.dumps({"frames":len(errors),"max_absolute_error":np.max(errors,axis=0).tolist()},indent=2)+"\n")
print("Exact split verified",np.max(errors,axis=0))
