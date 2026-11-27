"""T32 analysis runner matching scripts/board/nvidia_sb_pcm.c."""
from pathlib import Path

import numpy as np
import onnxruntime as ort

RATE = 16000
FFT = 510
HOP = 128
BINS = 256
SCALE = .33


def hann_window():
    return .5 - .5 * np.cos(2 * np.pi * np.arange(FFT) / FFT)


class OnnxEstimator:
    def __init__(self, model, threads):
        path = Path(model)
        if not path.is_file():
            raise FileNotFoundError(
                f"T32 ONNX model missing at {path}. Export it with "
                "scripts/npu/export_nvidia_sb_live.py --frames 32 and set NVIDIA_T32_ONNX."
            )
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.log_severity_level = 3
        self.session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        [input_info] = self.session.get_inputs()
        self.input_name = input_info.name
        self.shape = input_info.shape

    def __call__(self, spectrum):
        packed = np.stack((spectrum.real, spectrum.imag), axis=0)[None].astype(np.float32)
        return self.session.run(None, {self.input_name: packed})[0][0]


class Pipeline:
    def __init__(self, frames, hann, estimator, sigma=0.0):
        if sigma != 0:
            raise ValueError("The released deterministic estimator uses sigma=0")
        self.frames = frames
        self.window = (frames - 1) * HOP
        self.hann = hann
        self.estimator = estimator
        self.denominator = np.zeros(self.window)
        positions = np.arange(frames)[:, None] * HOP + np.arange(FFT) - FFT // 2
        self.positions = positions
        for start in np.arange(frames) * HOP - FFT // 2:
            a, b = max(0, -start), min(FFT, self.window - start)
            self.denominator[start + a:start + b] += hann[a:b] ** 2

    def enhance(self, pcm, count):
        peak = np.max(np.abs(pcm[:count].astype(np.int32))) if count else 0
        positions = self.positions
        frame = pcm[np.clip(positions, 0, self.window - 1)].astype(np.float64)
        frame *= ((positions >= 0) & (positions < count)) / (peak + 32768e-8)
        spectrum = np.fft.rfft(frame * self.hann, axis=1).T
        compressed = spectrum * SCALE / np.sqrt(np.maximum(np.abs(spectrum), 1e-30))
        estimate = self.estimator(compressed)
        predicted = (.0000331757874 * compressed +
                     .9999668 * (estimate[0] + 1j * estimate[1])) / SCALE
        restored = predicted * np.abs(predicted)
        decoded = np.fft.irfft(restored.T, FFT, axis=1)
        audio = np.zeros(self.window)
        for k, start in enumerate(np.arange(self.frames) * HOP - FFT // 2):
            a, b = max(0, -start), min(FFT, self.window - start)
            audio[start + a:start + b] += decoded[k, a:b] * self.hann[a:b]
        return audio / self.denominator * peak / 32768

    def process(self, pcm, history, overlap, dry_gain):
        fresh = self.window - history
        stride = fresh - overlap
        if history < 0 or stride <= 0 or overlap < 0 or overlap > stride:
            raise ValueError("Invalid history or overlap")
        pcm = np.asarray(pcm)
        if pcm.dtype != np.int16 or pcm.ndim != 1:
            raise ValueError("Expected mono PCM16 samples")
        if not 0 <= dry_gain <= 1:
            raise ValueError("Expected dry_gain in [0,1]")
        if len(pcm) == 0:
            return pcm.copy(), np.empty(0, dtype=np.float64), 0
        buffer = np.zeros(self.window, dtype=np.int16)
        position = min(fresh, len(pcm))
        buffer[history:history + position] = pcm[:position]
        got = position
        pending = np.zeros(overlap)
        pending_dry = np.zeros(overlap, dtype=np.int16)
        tail, blocks = 0, 0
        chunks, estimates = [], []

        def quantize(audio, dry):
            return np.clip(np.rint(audio * 32768 + dry_gain * dry.astype(np.float64)),
                           -32768, 32767).astype(np.int16)

        while got > 0:
            audio = self.enhance(buffer, history + got)
            output = audio[history:history + got].copy()
            if tail:
                mix = np.arange(tail) / (overlap - 1) if overlap > 1 else 0.0
                output[:tail] = pending[:tail] * (1 - mix) + output[:tail] * mix
            emit = min(got, stride)
            chunks.append(quantize(output[:emit], buffer[history:history + emit]))
            estimates.append(output[:emit].copy())
            tail = got - emit
            if tail:
                pending[:tail] = output[emit:emit + tail]
                pending_dry[:tail] = buffer[history + emit:history + emit + tail]
            blocks += 1
            if got < fresh:
                break
            buffer[:history + overlap] = buffer[stride:stride + history + overlap]
            new = min(stride, len(pcm) - position)
            buffer[history + overlap:history + overlap + new] = pcm[position:position + new]
            position += new
            if new == 0:
                break
            got = overlap + new
        if tail:
            chunks.append(quantize(pending[:tail], pending_dry[:tail]))
            estimates.append(pending[:tail].copy())
        result, predicted = np.concatenate(chunks), np.concatenate(estimates)
        if result.shape != pcm.shape or not np.isfinite(predicted).all():
            raise RuntimeError("Invalid reconstructed T32 audio")
        return result, predicted, blocks
