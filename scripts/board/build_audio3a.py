#!/usr/bin/env python3
"""Build the current DPDFNet2/RNNoise Audio3A chain without optional backends."""
import argparse,hashlib,json,os,subprocess
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('output',type=Path)
p.add_argument('--cc',default=os.environ.get('CC','cc'))
p.add_argument('--ort-include',required=True,type=Path)
p.add_argument('--ort-lib',required=True,type=Path)
p.add_argument('--cpu-flags',default='')
a=p.parse_args()
repo=Path(__file__).resolve().parents[2]
root=repo/'src/audio3a'
a.output.mkdir(parents=True,exist_ok=True)
(a.output/'bin').mkdir(exist_ok=True)
for path in [a.ort_include/'onnxruntime_c_api.h',a.ort_lib/'libonnxruntime.so']:
 if not path.is_file():p.error(f'Missing {path}')
names='audio_buffer fast_fft linear_aec bark_features neural_res_ans adaptive_agc transient_crossover spatial_dereverb online_wpe wav_file dpdfnet_ans celt_lpc denoise kiss_fft pitch rnn rnn_data'.split()
sources=[root/'tools/p2p_aec/p2p_dut_node.c',*[root/'src'/f'{n}.c' for n in names],*sorted((root/'src/webrtc_vad').glob('*.c'))]
args=[a.cc,'-O3','-std=gnu11','-Wall','-Wextra','-pthread',*a.cpu_flags.split(),
 '-I'+str(root/'include'),'-I'+str(root/'include/rnnoise_v1'),
 '-I'+str(root/'include/webrtc_vad'),'-I'+str(root/'src'),'-I'+str(a.ort_include),
 *map(str,sources),'-L'+str(a.ort_lib),'-lonnxruntime','-lm','-ldl','-lrt',
 '-Wl,-rpath,$ORIGIN/../lib','-o',str(a.output/'bin/p2p_dut_node')]
subprocess.run(args,check=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
metadata={'compiler':subprocess.check_output([a.cc,'--version'],text=True).splitlines()[0],
 'optional_denoisers':False,'sources':{str(x.relative_to(repo)):sha(x) for x in sources},
 'headers':{str(x.relative_to(repo)):sha(x) for x in sorted((root/'include').rglob('*.h'))},
 'binary_sha256':sha(a.output/'bin/p2p_dut_node'),'command':args}
(a.output/'BUILD.json').write_text(json.dumps(metadata,indent=2))
print(metadata['binary_sha256'])
