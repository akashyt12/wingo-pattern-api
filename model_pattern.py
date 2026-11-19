"""Pure NumPy inference for PatternNet-5M (no torch on Vercel)."""
import math
from pathlib import Path

import numpy as np

_ERF = np.vectorize(math.erf)


def gelu(x):
    return 0.5 * x * (1.0 + _ERF(x / math.sqrt(2.0)))


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


class PatternNetNP:
    def __init__(self, path=None):
        path = path or str(Path(__file__).resolve().parent / "weights.npz")
        self.w = np.load(path)

    def _linear(self, x, p):
        return x @ self.w[p + ".weight"].T + self.w[p + ".bias"]

    def _ln(self, x, p):
        m = x.mean(axis=-1, keepdims=True)
        v = ((x - m) ** 2).mean(axis=-1, keepdims=True)
        x = (x - m) / np.sqrt(v + 1e-5)
        return x * self.w[p + ".weight"] + self.w[p + ".bias"]

    def forward(self, x):
        x = np.asarray(x, dtype=np.float32)
        h = gelu(self._linear(x, "trunk.0"))
        h = self._ln(h, "trunk.2")
        h = gelu(self._linear(h, "trunk.4"))
        h = self._ln(h, "trunk.6")
        h = gelu(self._linear(h, "trunk.8"))
        h = self._ln(h, "trunk.10")
        return self._linear(h, "hs"), self._linear(h, "hc")

    def predict_proba(self, x):
        a, b = self.forward(x)
        return softmax(a, axis=-1), softmax(b, axis=-1)
