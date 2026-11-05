import os
import sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crm_model import CRMFilterV2, CRM2Core

ckpt = sys.argv[1]
out = sys.argv[2]
m = CRMFilterV2(ch=128).eval()
sd = torch.load(ckpt, map_location='cpu')
sd = sd['model'] if isinstance(sd, dict) and 'model' in sd else sd
m.load_state_dict({k.replace('module.', ''): v for k, v in sd.items()})
core = CRM2Core(m).eval()
x = torch.zeros(1, 2, 188, 257)
torch.onnx.export(core, x, out, input_names=['spec'], output_names=['enh'],
                  opset_version=17, do_constant_folding=True,
                  dynamic_axes={'spec': {2: 'T'}, 'enh': {2: 'T'}})
print('exported', out)
