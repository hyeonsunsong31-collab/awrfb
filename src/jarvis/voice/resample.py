"""Streaming polyphase resampler (Kaiser-windowed sinc), numpy only.

Used for mic/loopback audio (device rate -> 16 kHz) and TTS output (engine rate -> device rate).
Output sample n is sum_k h[p + k*L] * x[i0 - k] with p = (n*M) mod L, i0 = (n*M) div L for the
reduced ratio L/M = out_rate/in_rate, so a block costs one gather and one multiply-add per tap.
"""

from __future__ import annotations

from math import gcd

import numpy as np


class Resampler:
    def __init__(self, in_rate: int, out_rate: int, taps_per_phase: int = 24, beta: float = 8.0) -> None:
        g = gcd(int(in_rate), int(out_rate))
        self.in_rate, self.out_rate = int(in_rate), int(out_rate)
        self.L = self.out_rate // g  # upsampling factor
        self.M = self.in_rate // g  # downsampling factor
        # Taps per phase grow with the decimation factor so the anti-alias filter stays sharp.
        self.T = -(-taps_per_phase * max(self.L, self.M) // self.L)
        self.passthrough = self.L == self.M
        if self.passthrough:
            return
        n = self.L * self.T
        cutoff = 0.5 / max(self.L, self.M) * 0.94  # cycles per upsampled sample, just below Nyquist
        t = np.arange(n) - (n - 1) / 2.0
        h = 2.0 * cutoff * np.sinc(2.0 * cutoff * t) * np.kaiser(n, beta)
        h *= self.L / h.sum()  # zero-stuffing by L loses that much gain
        # phases[p, j] = h[p + (T-1-j)*L]: taps ordered to match a forward input window.
        self._phases = h.reshape(self.T, self.L).T[:, ::-1].astype(np.float32).copy()
        self.delay_out = (n - 1) / 2.0 / self.M  # group delay in output samples
        self.reset()

    def reset(self) -> None:
        if self.passthrough:
            return
        self._buf = np.zeros(self.T - 1, np.float32)
        self._buf_start = -(self.T - 1)  # absolute input index of _buf[0]
        self._n = 0  # next output index

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, np.float32).reshape(-1)
        if self.passthrough:
            return x.copy()
        self._buf = np.concatenate([self._buf, x])
        last = self._buf_start + len(self._buf) - 1
        n_end = (last * self.L + self.L - 1) // self.M + 1  # outputs whose newest input has arrived
        if n_end <= self._n:
            return np.zeros(0, np.float32)
        j = np.arange(self._n, n_end, dtype=np.int64) * self.M
        p = j % self.L
        start = j // self.L - (self.T - 1) - self._buf_start
        win = self._buf[start[:, None] + np.arange(self.T)[None, :]]
        y = np.einsum("nt,nt->n", win, self._phases[p])
        self._n = n_end
        keep = (self._n * self.M) // self.L - (self.T - 1) - self._buf_start
        if keep > 0:
            self._buf = self._buf[keep:]
            self._buf_start += keep
        return y.astype(np.float32)


def resample(x: np.ndarray, in_rate: int, out_rate: int) -> np.ndarray:
    """One-shot resampling with the filter delay removed (output length = round(len * out/in))."""
    x = np.asarray(x, np.float32).reshape(-1)
    if int(in_rate) == int(out_rate) or len(x) == 0:
        return x.copy()
    r = Resampler(in_rate, out_rate)
    want = round(len(x) * r.L / r.M)
    d = round(r.delay_out)
    y = r.process(np.concatenate([x, np.zeros(r.T * 2, np.float32)]))
    return y[d : d + want]


def to_mono(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float32)
    return x.mean(axis=1) if x.ndim == 2 else x.reshape(-1)


def float_to_int16(x: np.ndarray) -> np.ndarray:
    return (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16)


def int16_to_float(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, np.int16).astype(np.float32) / 32768.0


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if len(x) else 0.0
