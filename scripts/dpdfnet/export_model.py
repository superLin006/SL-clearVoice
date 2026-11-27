"""Export the validated DPDFNet2 RK3588 model and its FP32 normalization assets."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

SOURCE_SHA256 = "ce35d6025fc71df0ef10d1540e1b7916837bbfe5f6896deb744508d2cad487a9"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--rknn", action="store_true", help="Compile FP16 with RKNN Toolkit2.")
    args = parser.parse_args()
    if hashlib.sha256(args.source.read_bytes()).hexdigest() != SOURCE_SHA256:
        parser.error("The input is not the validated 16 kHz DPDFNet2 checkpoint.")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    script_dir = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix="dpdfnet-export-") as temp_dir:
        workspace = Path(temp_dir)
        shutil.copyfile(args.source, workspace / "original.onnx")
        for script in ("_export_core.py", "_fuse_linear.py", "_fuse_gru_gates.py", "_merge_gru_directions.py"):
            script_path = workspace / script
            shutil.copyfile(script_dir / script, script_path)
            subprocess.run([sys.executable, str(script_path)], check=True)
        subprocess.run(
            [sys.executable, str(workspace / "_fuse_linear.py"), "--cpu"], check=True
        )
        files = {
            "original.onnx": "dpdfnet2.onnx",
            "original-batched.onnx": "dpdfnet2-cpu.onnx",
            "core-merged-directions.onnx": "dpdfnet2-core.onnx",
            "initial-state.bin": "initial-state.bin",
            "erb-filterbank.bin": "erb-filterbank.bin",
            "export-validation.json": "split-validation.json",
            "batched-validation.json": "linear-validation.json",
            "gate-validation.json": "gate-validation.json",
            "cpu-validation.json": "cpu-validation.json",
            "merged-directions-validation.json": "direction-validation.json",
        }
        for original, name in files.items():
            shutil.copyfile(workspace / original, output / name)
    shutil.copyfile(script_dir.parents[1] / "models" / "dpdfnet" / "LICENSE", output / "LICENSE")
    if args.rknn:
        from rknn.api import RKNN

        runtime = RKNN(verbose=False)
        try:
            if runtime.config(target_platform="rk3588", optimization_level=3):
                raise RuntimeError("RKNN configuration failed.")
            if runtime.load_onnx(model=str(output / "dpdfnet2-core.onnx")):
                raise RuntimeError("RKNN ONNX import failed.")
            if runtime.build(do_quantization=False):
                raise RuntimeError("RKNN build failed.")
            if runtime.export_rknn(str(output / "dpdfnet2-core.rknn")):
                raise RuntimeError("RKNN export failed.")
        finally:
            runtime.release()
    manifest = {
        "source_sha256": SOURCE_SHA256,
        "sample_rate": 16000,
        "hop_samples": 160,
        "fft_samples": 320,
        "spectral_delay_samples": 640,
        "startup_hops": 5,
        "source_url": "https://huggingface.co/Ceva-IP/DPDFNet",
        "state_values": 45424,
        "feature_normalization": "CPU FP32",
        "neural_network": "RK3588 FP16, original weights",
        "transforms": ["batched grouped linear projections", "fused GRU-cell gate projections",
                       "merged frequency GRU directions with block-diagonal weights"],
        "files": {
            file.name: hashlib.sha256(file.read_bytes()).hexdigest()
            for file in sorted(output.iterdir())
            if file.is_file() and file.name not in ("manifest.json", "SHA256SUMS")
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "SHA256SUMS").write_text(
        "".join(f"{sha}  {name}\n" for name, sha in manifest["files"].items())
    )
    print(output)


if __name__ == "__main__":
    main()
