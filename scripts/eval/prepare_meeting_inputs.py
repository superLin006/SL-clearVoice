#!/usr/bin/env python3
"""Copy six raw meeting WAVs; derive recorded gain for quiet cases 3/4.

Only channels 0..7 are amplified to joint RMS -36 dBFS. References 8/9 remain
sample-identical. Source files are never changed; derived files and their hashes
are written to the explicit output directory.
"""
import argparse,hashlib,json,shutil
from pathlib import Path
import numpy as np
import soundfile as sf

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def prepare(source, output):
    source, output = source.resolve(), output.resolve()
    if output == source or source in output.parents:
        raise ValueError("Output must be outside the raw input directory")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for case in range(1,7):
        name = f"meeting_{case}.raw10ch.wav"
        path, target = source/name, output/name
        if target.exists():
            raise FileExistsError(target)
        audio, rate = sf.read(path, dtype="int16", always_2d=True)
        if rate != 16000 or audio.shape[1] != 10:
            raise ValueError(f"{path}: expected 10-channel 16kHz PCM16")
        gain_db = 0.
        if case in (3,4):
            mic = audio[:,:8].astype(np.float64)/32768.
            rms = np.sqrt(np.mean(mic*mic))
            if rms == 0:
                raise ValueError(f"{path}: silent microphone channels")
            gain_db = float(-36 - 20*np.log10(rms))
            # Match the float-to-PCM16 floor used for the qualified derived WAVs.
            scaled = np.floor(audio[:,:8].astype(np.float64)*10**(gain_db/20))
            if scaled.max() > 32767 or scaled.min() < -32768:
                raise ValueError(f"{path}: derived gain would clip")
            audio[:,:8] = scaled.astype(np.int16)
            sf.write(target, audio, rate, subtype="PCM_16")
        else:
            shutil.copyfile(path, target)
        manifest[path.stem] = dict(source=str(path), source_sha256=sha(path),
            derived_sha256=sha(target),gain_db_mic_channels=gain_db,
            samples=len(audio),sample_rate=rate,channels=10)
    (output/"input-manifest.json").write_text(json.dumps(manifest,indent=2))
    return manifest

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = prepare(args.source,args.output)
    print(f"Prepared {len(result)} recordings; provenance: {args.output/'input-manifest.json'}")
