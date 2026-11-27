#!/usr/bin/env python3
"""Exercise the built chain and real DPDF with a deterministic pipe worker.

The identity worker validates dispatch, EOF and ownership; it does not validate
NVIDIA quality or NPU speed. Real-board tests cover those separately.
"""
import argparse,os,subprocess,tempfile,wave
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser()
p.add_argument('binary',type=Path);p.add_argument('--model',type=Path,required=True)
a=p.parse_args()
with tempfile.TemporaryDirectory(prefix='sl-chain-check-') as tmp:
 d=Path(tmp);worker=d/'identity'
 worker.write_text('#!/usr/bin/env python3\nimport os\nwhile True:\n x=os.read(0,4096)\n if not x:break\n os.write(1,x)\n')
 worker.chmod(0o755);(d/'dummy-t32.rknn').touch()
 n=25600;t=np.arange(n)/16000
 mono=(800*np.sin(2*np.pi*211*t)*(np.sin(2*np.pi*2*t)>0)).astype('<i2')
 audio=np.zeros((n,10),dtype='<i2');audio[:,:8]=mono[:,None]
 with wave.open(str(d/'input.wav'),'wb') as f:
  f.setparams((10,2,16000,0,'NONE',''));f.writeframes(audio.tobytes())
 base=[str(a.binary),'--wpe','--no-dereverb','--ans','dpdfnet','--dpdfnet-model',str(a.model),'--mic-pregain','1',
       '--nvidia','--nvidia-bin',str(worker),'--nvidia-rknn',str(d/'dummy-t32.rknn')]
 for name in ['pre','post','bypass','restore_agc']:
  (d/name).mkdir()
  cmd=base+['--offline-wav',str(d/'input.wav'),str(d/name/'output.wav'),'--dump-stages',str(d/name)]
  if name=='post':cmd+=['--denoise-after-nvidia']
  if name=='bypass':cmd+=['--bypass-post-nvidia-agc']
  if name=='restore_agc':cmd+=['--bypass-post-nvidia-agc','--post-nvidia-agc']
  r=subprocess.run(cmd,capture_output=True,text=True,timeout=20)
  assert r.returncode==0,r.stderr
  with wave.open(str(d/name/'output.wav')) as f:assert f.getnframes()==n
 assert (d/'pre/output.wav').read_bytes()==(d/'restore_agc/output.wav').read_bytes(),'AGC restore differs'
 for stage in ['00_raw_input_avg.wav','04_spatial_fusion.wav','07_transient_xover.wav']:
  assert (d/'pre'/stage).read_bytes()==(d/'post'/stage).read_bytes(),stage
 with wave.open(str(d/'bypass/output.wav')) as f, wave.open(str(d/'bypass/11_nvidia_ode_raw.wav')) as g:
  assert f.readframes(n)==g.readframes(n),'AGC bypass changed samples'
 r=subprocess.run([str(a.binary),'--denoise-after-nvidia','--ans','dpdfnet'],capture_output=True,timeout=3)
 assert r.returncode==2,r.stderr
 bad=base.copy();bad[bad.index('--dpdfnet-model')+1]=str(d/'missing.onnx')
 r=subprocess.run(bad+['--offline-wav',str(d/'input.wav'),str(d/'bad.wav')],capture_output=True,timeout=5)
 assert r.returncode!=0
print('PASS: pre/post dispatch, equal front ends, exact frame count, invalid mode and missing model')
