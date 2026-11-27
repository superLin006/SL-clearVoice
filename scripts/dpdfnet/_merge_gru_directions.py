"""Merge each frequency GRU's directions using block-diagonal original weights.

The explicit batch layout around the merged GRU is required by RKNN. Validate
the mathematical rewrite with FP32 recurrent frames before NPU compilation.
"""
from pathlib import Path
import json
import numpy as np
import onnx
from onnx import helper,numpy_helper,shape_inference
import onnxruntime as ort
ROOT=Path(__file__).resolve().parent
m=shape_inference.infer_shapes(onnx.load(ROOT/'core-gates.onnx'))
arrays={x.name:numpy_helper.to_array(x) for x in m.graph.initializer}
producers={o:n for n in m.graph.node for o in n.output}
consumers={v for n in m.graph.node for v in n.input}
shapes={v.name:[d.dim_value for d in v.type.tensor_type.shape.dim]
        for v in [*m.graph.input,*m.graph.value_info,*m.graph.output]}
nodes=[];report=[]
def init(name,array):
    m.graph.initializer.append(numpy_helper.from_array(np.asarray(array),name))
    return name
for node in m.graph.node:
    if node.op_type!='GRU':
        nodes.append(node);continue
    a={v.name:helper.get_attribute_value(v) for v in node.attribute}
    assert a['direction']==b'bidirectional'
    assert len(node.output)<2 or node.output[1] not in consumers
    if len(node.input)>4:assert not node.input[4]
    h0=producers[node.input[5]]
    assert h0.op_type=='Expand'
    constant=producers[h0.input[0]]
    assert constant.op_type=='Constant'
    value=numpy_helper.to_array(next(x.t for x in constant.attribute if x.name=='value'))
    assert np.all(value==0)
    w,r,b=[arrays[node.input[i]] for i in (1,2,3)]
    hidden=int(a['hidden_size']);width=w.shape[2]
    seq=shapes[node.input[0]][0]
    assert hidden==width==64 and seq in (8,48)
    out_hidden=2*hidden
    nw=np.zeros((1,3*out_hidden,2*width),np.float32)
    nr=np.zeros((1,3*out_hidden,out_hidden),np.float32)
    nb=np.zeros((1,6*out_hidden),np.float32)
    for direction in range(2):
        for gate in range(3):
            row=slice(gate*out_hidden+direction*hidden,gate*out_hidden+(direction+1)*hidden)
            nw[0,row,direction*width:(direction+1)*width]=w[direction,gate*hidden:(gate+1)*hidden]
            nr[0,row,direction*hidden:(direction+1)*hidden]=r[direction,gate*hidden:(gate+1)*hidden]
            nb[0,row]=b[direction,gate*hidden:(gate+1)*hidden]
            start=3*out_hidden+gate*out_hidden+direction*hidden
            nb[0,start:start+hidden]=b[direction,3*hidden+gate*hidden:3*hidden+(gate+1)*hidden]
    p=node.name+'/merged_directions'
    rev=init(p+'_reverse',np.arange(seq-1,-1,-1,dtype=np.int64))
    weights=init(p+'_w',nw);recurrent=init(p+'_r',nr);bias=init(p+'_b',nb)
    sections=init(p+'_sections',np.array([hidden,hidden],np.int64))
    initial=init(p+'_initial',np.zeros((1,1,out_hidden),np.float32))
    attrs={'direction':'forward','hidden_size':out_hidden,
           'linear_before_reset':a.get('linear_before_reset',0)}
    if 'clip' in a:attrs['clip']=a['clip']
    if 'activations' in a:
        acts=a['activations'];assert acts[:2]==acts[2:]
        attrs['activations']=acts[:2]

    in_shape=init(p+'_in_shape',np.array([1,seq,out_hidden],np.int64))
    split_shape=init(p+'_split_shape',np.array([1,seq,1,hidden],np.int64))
    nodes.extend([
        helper.make_node('Transpose',[node.input[0]],[p+'_batch_input'],name=p+'_input_layout',perm=[1,0,2]),
        helper.make_node('Gather',[p+'_batch_input',rev],[p+'_reverse_input'],name=p+'_gather_input',axis=1),
        helper.make_node('Concat',[p+'_batch_input',p+'_reverse_input'],[p+'_input_batch'],name=p+'_concat_input',axis=2),
        helper.make_node('Transpose',[p+'_input_batch'],[p+'_input'],name=p+'_sequence_layout',perm=[1,0,2]),
        helper.make_node('GRU',[p+'_input',weights,recurrent,bias,'',initial],[p+'_y',p+'_last'],name=p+'_gru',**attrs),
        helper.make_node('Transpose',[p+'_y'],[p+'_y_batch'],name=p+'_output_layout',perm=[2,0,1,3]),
        helper.make_node('Reshape',[p+'_y_batch',in_shape],[p+'_y_flat'],name=p+'_output_flat'),
        helper.make_node('Split',[p+'_y_flat',sections],[p+'_fwd',p+'_rev'],name=p+'_split',axis=2),
        helper.make_node('Gather',[p+'_rev',rev],[p+'_rev_aligned'],name=p+'_gather_output',axis=1),
        helper.make_node('Reshape',[p+'_fwd',split_shape],[p+'_fwd4'],name=p+'_fwd4_shape'),
        helper.make_node('Reshape',[p+'_rev_aligned',split_shape],[p+'_rev4'],name=p+'_rev4_shape'),
        helper.make_node('Transpose',[p+'_fwd4'],[p+'_fwd_seq'],name=p+'_fwd_seq',perm=[1,2,0,3]),
        helper.make_node('Transpose',[p+'_rev4'],[p+'_rev_seq'],name=p+'_rev_seq',perm=[1,2,0,3]),
        helper.make_node('Concat',[p+'_fwd_seq',p+'_rev_seq'],[node.output[0]],name=p+'_concat_output',axis=1),
    ])
    report.append({'name':node.name,'sequence':seq,'hidden_before':hidden,'hidden_after':out_hidden})
del m.graph.node[:];m.graph.node.extend(nodes)
needed={o.name for o in m.graph.output};live=[]
for node in reversed(m.graph.node):
    if any(v in needed for v in node.output):
        live.append(node);needed.update(node.input)
del m.graph.node[:];m.graph.node.extend(reversed(live))
constants=[x for x in m.graph.initializer if x.name in needed]
del m.graph.initializer[:];m.graph.initializer.extend(constants)
del m.graph.value_info[:]
onnx.checker.check_model(m)
onnx.save(m,ROOT/'core-merged-directions.onnx')
options=ort.SessionOptions();options.intra_op_num_threads=1
ref=ort.InferenceSession(str(ROOT/'core-gates.onnx'),options)
test=ort.InferenceSession(str(ROOT/'core-merged-directions.onnx'),options)
pre=ort.InferenceSession(str(ROOT/'preprocess.onnx'),options)
a=np.fromfile(ROOT/'initial-state.bin',np.float32);b=a.copy()
rng=np.random.default_rng(47);worst=np.zeros(2)
for frame in range(80):
    x=rng.normal(0,.8,(1,1,161,2)).astype(np.float32)
    if frame<4:x*=0
    ef,sf,es,ss=pre.run(None,{'spec':x,'state_in':a})
    a[:128]=b[:128]=np.r_[es,ss]
    inputs={'spec':x,'state_in':a,'erb_feat':ef,'spec_feat':sf}
    y,a=ref.run(None,inputs)
    inputs['state_in']=b
    z,b=test.run(None,inputs)
    worst=np.maximum(worst,[np.max(abs(y-z)),np.max(abs(a-b))])
assert worst.max()<2e-4,worst
(ROOT/'merged-directions-validation.json').write_text(json.dumps({
    'transforms':report,'frames':80,'max_absolute_error':worst.tolist()},indent=2)+'\n')
print('GRU direction merge verified',worst.tolist(),flush=True)
