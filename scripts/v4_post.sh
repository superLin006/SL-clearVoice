#!/bin/bash
# exact v4 post chain recovered from 2026-10-30 session
# usage: v4_post.sh input.wav output.wav
set -euo pipefail
FF=${FF:-ffmpeg}
AF='highpass=f=150:poles=2,equalizer=f=300:t=q:w=1.2:g=0.5,equalizer=f=1200:t=q:w=1.0:g=6.0,equalizer=f=2200:t=q:w=1.0:g=2.0,equalizer=f=3200:t=q:w=1.0:g=4.0,equalizer=f=5000:t=q:w=1.0:g=-1.0,lowpass=f=6000:poles=2,dynaudnorm=f=300:g=7:p=0.95:m=10,loudnorm=I=-16.9:TP=-1.3:LRA=4.2,agate=threshold=0.02:ratio=2:range=0.02:attack=10:release=250'
"$FF" -y -v error -i "$1" -af "$AF" -ar 16000 "$2"
