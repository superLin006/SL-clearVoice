#!/usr/bin/env python3
"""Paired Audio3A runs: equal-loudness audio, spectra, DNSMOS and alignment."""
import argparse,json,re,subprocess
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy import signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import onnxruntime as ort
from speechmos import dnsmos
from audio_metrics import score as metric_score
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('root',type=Path)
p.add_argument('--output',required=True,type=Path)
p.add_argument('--ffmpeg',required=True)
p.add_argument('--cases',nargs='+',type=int,default=list(range(1,7)))
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
o=ort.SessionOptions();o.intra_op_num_threads=2;o.inter_op_num_threads=1
sess=ort.InferenceSession(str(Path(dnsmos.__file__).parent/'dnsmos_models/sig_bak_ovr.onnx'),o,providers=['CPUExecutionProvider'])
font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
if Path(font).exists():font_manager.fontManager.addfont(font);plt.rcParams['font.family']='Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus']=False
def load(p):
 x,sr=sf.read(p);assert sr==16000 and x.ndim==1;return x
def levels(p):
 r=subprocess.run([a.ffmpeg,'-nostdin','-i',str(p),'-af','loudnorm=print_format=json','-f','null','-'],capture_output=True,text=True,check=True)
 d=json.loads(re.findall(r'\{[^{}]*"input_i"[^{}]*\}',r.stderr)[-1]);return float(d['input_i']),float(d['input_tp'])
def dns(x):
 return metric_score(x,sess)
def delay(x,y):
 # Median waveform correlation across five speech-containing regions; this is
 # an alignment estimate, not a dereverberation or quality score.
 n=min(len(x),len(y));values=[]
 sos=signal.butter(3,[180,3600],btype='bandpass',fs=16000,output='sos')
 for at in np.linspace(16000,max(16000,n-4*16000),5,dtype=int):
  u=signal.sosfilt(sos,x[at:at+3*16000]);v=signal.sosfilt(sos,y[at:at+3*16000])
  if len(u)<16000 or np.mean(u*u)<1e-12:continue
  c=signal.correlate(v,u,mode='full',method='fft')
  l=signal.correlation_lags(len(v),len(u))
  keep=abs(l)<=3200
  values.append(int(l[keep][np.argmax(abs(c[keep]))]))
 return int(np.median(values)) if values else 0
report={}
report_file=a.output/'report.json'
if report_file.exists():report=json.loads(report_file.read_text())
for k in a.cases:
 dirs={o:a.root/f'm{k}-{o}' for o in ['pre','post']}
 if not all((d/'output.wav').exists() for d in dirs.values()):continue
 # Front end must match byte-for-byte before interpreting model order.
 for stage in ['00_raw_input_avg','04_spatial_fusion','07_transient_xover']:
  assert (dirs['pre']/(stage+'.wav')).read_bytes()==(dirs['post']/(stage+'.wav')).read_bytes(),(k,stage)
 reference=load(dirs['pre']/'07_transient_xover.wav')
 tracks={o:load(d/'output.wav') for o,d in dirs.items()}
 n=len(reference);assert all(len(x)==n for x in tracks.values())
 stats={};aligned={}
 for name,x in tracks.items():
  lag=delay(reference,x)
  aligned[name]=x[max(0,lag):min(n,n+lag)]
  if lag<0:aligned[name]=np.pad(aligned[name],(-lag,0))
  else:aligned[name]=np.pad(aligned[name],(0,lag))
  blocks=n//320
  power=np.mean(reference[:blocks*320].reshape(blocks,320)**2,axis=1)
  audible=power>10**(-85/10)
  q=np.quantile(power[audible],.2) if audible.any() else 0
  hi=np.quantile(power[audible],.65) if audible.any() else 0
  ey=np.mean(aligned[name][:blocks*320].reshape(blocks,320)**2,axis=1)
  quiet=(power<=q)&audible
  speech=(power>=hi)&audible
  quiet[:50]=False;speech[:50]=False
  contrast=float(10*np.log10((np.mean(ey[quiet])+1e-24)/(np.mean(ey[speech])+1e-24))) if quiet.any() and speech.any() else None
  lu,tp=levels(dirs[name]/'output.wav')
  stats[name]={'dnsmos':dns(x[16000:-8000]),'lufs':lu,'true_peak':tp,'alignment_samples':lag,
    'rms_dbfs':float(10*np.log10(np.mean(x*x)+1e-24)), 'quiet_to_active_db':contrast}
 valid_start=max(0,*[-v['alignment_samples'] for v in stats.values()])
 valid_end=min(n,*[n-v['alignment_samples'] for v in stats.values()])
 aligned={name:x[valid_start:valid_end] for name,x in aligned.items()}
 target=min(-23.,*(v['lufs']-v['true_peak']-1.2 for v in stats.values()))
 for name,x in aligned.items():
  sf.write(a.output/f'm{k}-{name}-matched.wav',x*10**((target-stats[name]['lufs'])/20),16000,subtype='PCM_16')
 fig,axes=plt.subplots(2,1,sharex=True,figsize=(14,5.5))
 fig.subplots_adjust(left=.06,right=.93,top=.88,bottom=.1,hspace=.48)
 for ax,(name,x) in zip(axes,aligned.items()):
  y=x*10**((target-stats[name]['lufs'])/20)
  f,t,z=signal.stft(y,16000,nperseg=512,noverlap=256,boundary=None)
  m=ax.pcolormesh(t,f/1000,20*np.log10(np.maximum(abs(z),1e-10)),vmin=-95,vmax=-35,cmap='magma',shading='auto',rasterized=True)
  ax.set_title('DPDFNet2 → NVIDIA' if name=='pre' else 'NVIDIA → DPDFNet2',loc='left');ax.set_ylabel('kHz')
 axes[-1].set_xlabel('秒');axes[-1].set_xlim(0,(valid_end-valid_start)/16000)
 fig.suptitle(f'meeting_{k} · 完整链路 · 同响度 {target:.1f} LUFS')
 fig.colorbar(m,cax=fig.add_axes((.95,.1,.012,.78))).set_label('dBFS')
 fig.savefig(a.output/f'm{k}-spectrogram.png',dpi=130);plt.close(fig)
 report[str(k)]={'samples':n,'seconds':n/16000,'target_lufs':target,'valid_samples':valid_end-valid_start,'tracks':stats}
 report_file.write_text(json.dumps(report,ensure_ascii=False,indent=2))
 print(k,stats,flush=True)
