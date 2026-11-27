"""Shared DNSMOS settings for paired chain experiments (no clean reference)."""
from pathlib import Path
import numpy as np
import onnxruntime as ort
from speechmos import dnsmos
WINDOW=144160
STEP=32000
def session():
 options=ort.SessionOptions()
 options.intra_op_num_threads=2
 options.inter_op_num_threads=1
 return ort.InferenceSession(str(Path(dnsmos.__file__).parent/'dnsmos_models/sig_bak_ovr.onnx'),options,providers=['CPUExecutionProvider'])
def score(x,sess):
 x=np.asarray(x,dtype=np.float32).copy();x-=x.mean()
 if np.mean(x*x)<1e-15:return None
 x=np.clip(x*.03/np.sqrt(np.mean(x*x)),-1,1)
 while len(x)<WINDOW:x=np.r_[x,x]
 rows=[]
 for k in range(0,len(x)-WINDOW+1,STEP):
  values=sess.run(None,{'input_1':x[k:k+WINDOW][None]})[0][0]
  rows.append(dnsmos.DNSMOS.get_polyfit_val(None,*values,False))
 return dict(zip(['sig','bak','ovrl'],map(float,np.mean(rows,axis=0))))
