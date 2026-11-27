"""Minimal NumPy generative adversarial network (used by the GAN-ACO baseline).

Generator and discriminator are MLPs with hidden layers (128, 64), LeakyReLU
activations and Adam (lr = 1e-3, batch 64), matching the configuration in the
reference article (its Table 4).  The generator maps Gaussian noise to a vector
of k cluster centroids in normalised coordinates.
"""
from __future__ import annotations

import numpy as np


class _MLP:
    def __init__(self, sizes, rng, out_sigmoid: bool):
        self.W = [rng.normal(0.0, np.sqrt(2.0 / a), (a, b)) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b) for b in sizes[1:]]
        self.out_sigmoid = out_sigmoid
        self.m = [np.zeros_like(p) for p in self.W + self.b]
        self.v = [np.zeros_like(p) for p in self.W + self.b]
        self.t = 0

    def forward(self, x):
        self.acts = [x]
        self.pre = []
        h = x
        L = len(self.W)
        for k in range(L):
            z = h @ self.W[k] + self.b[k]
            self.pre.append(z)
            if k < L - 1:
                h = np.where(z > 0, z, 0.2 * z)
            else:
                h = 1.0 / (1.0 + np.exp(-z)) if self.out_sigmoid else z
            self.acts.append(h)
        return h

    def backward(self, g_out):
        """g_out: gradient w.r.t. the pre-activation of the last layer if the
        output is linear, or w.r.t. the output if sigmoid."""
        L = len(self.W)
        gW, gb = [None] * L, [None] * L
        g = g_out
        if self.out_sigmoid:
            s = self.acts[-1]
            g = g * s * (1.0 - s)
        for k in range(L - 1, -1, -1):
            gW[k] = self.acts[k].T @ g
            gb[k] = g.sum(axis=0)
            g = g @ self.W[k].T
            if k > 0:
                g = g * np.where(self.pre[k - 1] > 0, 1.0, 0.2)
        return gW, gb, g

    def step(self, gW, gb, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        params = self.W + self.b
        grads = gW + gb
        for k, (p, g) in enumerate(zip(params, grads)):
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * g * g
            mh = self.m[k] / (1 - b1 ** self.t)
            vh = self.v[k] / (1 - b2 ** self.t)
            p -= lr * mh / (np.sqrt(vh) + eps)


class CentroidGAN:
    def __init__(self, dim: int, rng: np.random.Generator, zdim: int = 8,
                 hidden=(128, 64), lr: float = 1e-3, batch: int = 64):
        self.rng = rng
        self.zdim = zdim
        self.lr = lr
        self.batch = batch
        self.G = _MLP([zdim, *hidden, dim], rng, out_sigmoid=True)
        self.D = _MLP([dim, *hidden, 1], rng, out_sigmoid=False)

    def train(self, real: np.ndarray, iters: int) -> None:
        rng, bs = self.rng, self.batch
        for _ in range(iters):
            # discriminator step: maximise log D(x) + log(1 - D(G(z)))
            xr = real[rng.integers(0, len(real), bs)]
            xf = self.G.forward(rng.normal(size=(bs, self.zdim)))
            lr_ = self.D.forward(xr)
            pr = 1.0 / (1.0 + np.exp(-lr_))
            gW1, gb1, _ = self.D.backward((pr - 1.0) / bs)
            lf = self.D.forward(xf)
            pf = 1.0 / (1.0 + np.exp(-lf))
            gW2, gb2, _ = self.D.backward(pf / bs)
            self.D.step([a + b for a, b in zip(gW1, gW2)], [a + b for a, b in zip(gb1, gb2)], self.lr)
            # generator step (non-saturating loss): maximise log D(G(z))
            z = rng.normal(size=(bs, self.zdim))
            xf = self.G.forward(z)
            lf = self.D.forward(xf)
            pf = 1.0 / (1.0 + np.exp(-lf))
            _, _, gx = self.D.backward((pf - 1.0) / bs)
            gW, gb, _ = self.G.backward(gx)
            self.G.step(gW, gb, self.lr)

    def sample(self, n: int = 1) -> np.ndarray:
        return self.G.forward(self.rng.normal(size=(n, self.zdim)))
