"""DPDFNet transform and FIFO matching src/audio3a/src/dpdfnet_ans.c.

The default 256-sample input cadence retains the board's initial 160-sample FIFO
and STFT delay. Returned length equals input length; pending EOF audio is not
silently flushed or realigned. This is an evaluation runner, not a latency test.
"""
from collections import deque
from pathlib import Path
import numpy as np
import onnxruntime as ort

DEFAULT_MODEL = Path(__file__).resolve().parents[2] / "models/dpdfnet/dpdfnet2.onnx"
HOP, WINDOW = 160, 320


class DpdfRunner:
    def __init__(self, model=DEFAULT_MODEL, threads=1):
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model), options, providers=["CPUExecutionProvider"])
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        self.input_names = [v.name for v in inputs]
        self.output_names = [v.name for v in outputs]
        metadata = self.session.get_modelmeta().custom_metadata_map
        size = inputs[1].shape[-1]
        if not isinstance(size, int) or size <= 0:
            size = int(metadata["state_size"])
        self.initial = np.zeros(size, dtype=np.float32)
        erb_size = int(metadata.get("erb_norm_state_size", "32"))
        spec_size = int(metadata.get("spec_norm_state_size", "96"))
        for key, start, count in (("erb_norm_init", 0, erb_size),
                                  ("spec_norm_init", erb_size, spec_size)):
            if key in metadata and start < size:
                values = np.fromstring(metadata[key], sep=",", dtype=np.float32)
                count = min(count, size - start, len(values))
                self.initial[start:start+count] = values[:count]
        n = np.arange(WINDOW, dtype=np.float32)
        sine = np.sin(np.float32(np.pi) * (n + np.float32(.5)) / np.float32(WINDOW))
        self.window = np.sin(np.float32(.5*np.pi) * sine * sine)
        self.reset()

    def reset(self):
        self.state = self.initial.copy()
        self.history = np.zeros(HOP, dtype=np.float32)
        self.ola = np.zeros(HOP, dtype=np.float32)
        self.input_fifo = deque()
        self.output_fifo = deque(np.zeros(HOP, dtype=np.float32))

    def _hop(self, current):
        z = np.fft.rfft(np.r_[self.history, current] * self.window)
        self.history = current.copy()
        spec = np.stack([z.real, z.imag], axis=-1)[None, None].astype(np.float32)
        result, self.state = self.session.run(
            self.output_names, dict(zip(self.input_names, [spec, self.state])))
        spectrum = result[0, 0, :, 0] + 1j * result[0, 0, :, 1]
        y = np.fft.irfft(spectrum, n=WINDOW).astype(np.float32) * self.window
        output = self.ola + y[:HOP]
        self.ola = y[HOP:].copy()
        return output

    def process(self, audio, pcm16=False):
        """Process from the current state, using board-sized <=256 sample calls."""
        x = np.asarray(audio)
        if x.ndim != 1:
            raise ValueError("Expected mono audio at 16 kHz")
        if np.issubdtype(x.dtype, np.integer):
            raise ValueError("Expected normalized floating-point audio")
        if pcm16:
            x = np.clip(np.rint(x * 32768), -32768, 32767).astype(np.int16)
            x = x.astype(np.float32) / np.float32(32768)
        else:
            x = x.astype(np.float32)
        output = np.zeros(len(x), dtype=np.float32)
        for at in range(0, len(x), 256):
            count = min(256, len(x)-at)
            self.input_fifo.extend(x[at:at+count])
            while len(self.input_fifo) >= HOP:
                hop = np.fromiter((self.input_fifo.popleft() for _ in range(HOP)),
                                  dtype=np.float32, count=HOP)
                self.output_fifo.extend(self._hop(hop))
            take = min(count, len(self.output_fifo))
            output[at:at+take] = np.fromiter(
                (self.output_fifo.popleft() for _ in range(take)), dtype=np.float32, count=take)
        if pcm16:
            output = np.clip(np.rint(output * 32768), -32768, 32767).astype(np.int16).astype(np.float32)/32768
        return output


def process(audio, model=DEFAULT_MODEL, threads=1):
    return DpdfRunner(model, threads).process(audio, pcm16=True)
