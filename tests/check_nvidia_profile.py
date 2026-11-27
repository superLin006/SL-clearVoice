#!/usr/bin/env python3
"""Verify model-name-independent profile dispatch and invalid argument rejection."""
import argparse,json,os,subprocess,tempfile,wave
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('binary',type=Path);p.add_argument('--model',type=Path,required=True)
a=p.parse_args()
with tempfile.TemporaryDirectory(prefix='sl-profile-check-') as tmp:
 d=Path(tmp);worker=d/'worker';model=d/'opaque.rknn';model.touch()
 worker.write_text("""#!/usr/bin/env python3
import json,os,sys
with open(os.environ['PROFILE_TEST_LOG'],'w') as f:
 json.dump({'args':sys.argv[2:],'frames':os.environ['SB_EXPECT_FRAMES']},f)
while True:
 x=os.read(0,4096)
 if not x:break
 os.write(1,x)
""")
 worker.chmod(0o755)
 x=np.zeros((1024,10),dtype='<i2')
 with wave.open(str(d/'input.wav'),'wb') as f:
  f.setparams((10,2,16000,0,'NONE',''));f.writeframes(x.tobytes())
 base=[str(a.binary),'--no-dereverb','--ans','dpdfnet','--dpdfnet-model',str(a.model),
       '--nvidia','--nvidia-bin',str(worker),'--nvidia-rknn',str(model)]
 for name,options,expected in [
  ('default',[],(['48','48'],'32')),
  ('history64',['--nvidia-history-ms','64'],(['48','64'],'32')),
  ('history80',['--nvidia-history-ms','80'],(['48','80'],'32')),
  ('t16',['--nvidia-frames','16'],(['8','16'],'16'))]:
  log=d/f'{name}.json'
  env=dict(os.environ,PROFILE_TEST_LOG=str(log))
  r=subprocess.run(base+options+['--offline-wav',str(d/'input.wav'),str(d/f'{name}.wav')],
                   env=env,capture_output=True,timeout=15)
  assert r.returncode==0,(name,r.stderr)
  actual=json.loads(log.read_text())
  assert actual==dict(args=expected[0],frames=expected[1]),(name,actual)
 for options in [
  ['--nvidia-frames','24'],['--nvidia-frames'],['--nvidia-history-ms','oops'],
  ['--nvidia-history-ms','11'],['--nvidia-history-ms','248'],
  ['--nvidia-overlap-ms','128'],['--nvidia-overlap-ms','-8']]:
  r=subprocess.run([str(a.binary),*options],capture_output=True,timeout=3)
  assert r.returncode==2,(options,r.returncode,r.stderr)
print('PASS: renamed model, explicit T16/T32 geometry, worker frame expectation, invalid options')
