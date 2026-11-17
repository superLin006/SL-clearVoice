"""Build an FP16 NVIDIA SB estimator for RK3588."""
import argparse
from pathlib import Path

from rknn.api import RKNN


def checked(status, operation):
    if status != 0:
        raise RuntimeError(f"{operation} returned {status}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    args = parser.parse_args()
    rknn = RKNN()
    try:
        checked(rknn.config(target_platform="rk3588", optimization_level=3), "config")
        checked(rknn.load_onnx(model=str(args.model)), "load")
        checked(rknn.build(do_quantization=False), "build")
        checked(rknn.export_rknn(str(args.model.with_suffix(".rknn"))), "export")
    finally:
        rknn.release()


if __name__ == "__main__":
    main()
