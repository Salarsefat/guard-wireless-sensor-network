"""Re-implementation of GAN-ACO (Sefati et al., Mathematics 2025, 13, 597).

Clustering: a GAN (generator/discriminator MLPs with 128 and 64 hidden units,
Adam, lr 1e-3, batch 64; the article's Table 4) is trained each round and its
generated centroids partition the nodes.  The article does not define the
"real" cluster data; we use centroid sets of randomly initialised k-means runs
(assumption A1 in docs/BASELINES.md).  Cluster heads maximise the article's
Eq. (13): S = wE*E/Emax - wD*d_BS/d_max - wC*|x - c_k|/d_avg.

Routing: ACO next-hop probability proportional to
tau^aT * (E_j/E0)^aE * PRR^aB * (1/(1+Q_j))^aL * (1/(1+H_j))^aS
with (aE, aB, aL, aS, aT) = (1.5, 1.2, 0.8, 1.0, 1.0) as listed in the article's
Table 3 (energy, bandwidth, latency, steps, pheromone).  Pheromone is
initialised to 1/(d + eps), evaporates with rho = 0.1 per second and is
reinforced by Q/L after each acknowledged transmission (Q = 0.5).  Links whose
congestion factor (delivered frames / link capacity) exceeds 0.7 are excluded.
Candidates are restricted to neighbours closer to the sink (loop avoidance, not
specified in the article).
"""
from __future__ import annotations

import numpy as np

from ..clustering import alive_sensors, assign_to_heads, kmeans
from ..gan import CentroidGAN
from ..sim import COLLECT
from .base import Protocol


class GANACO(Protocol):
    name = "ganaco"
    defaults = {"k": 12, "Tc": 30.0, "rho": 0.1, "Q": 0.5, "aE": 1.5, "aB": 1.2, "aL": 0.8,
                "aS": 1.0, "aT": 1.0, "wE": 1.5, "wD": 1.2, "wC": 0.8, "cf_thr": 0.7,
                "gan_iters": 200, "n_real": 64}

    def init_state(self):
        sim = self.sim
        Rr = sim.top.R
        self.tau = [{j: Rr / (sim.dist[i][j] + 1.0) for j in sim.nbr[i]} for i in range(sim.N)]
        self.succ: dict[int, int] = {}
        self.cf: dict[int, float] = {}
        self.E0 = sim.scen["E0"]

    def compute_clusters(self):
        sim, p = self.sim, self.p
        rng = sim.np_rng
        nodes = alive_sensors(sim)
        if not nodes:
            return [], {}
        A = sim.scen["area"]
        pts = np.array([sim.pos[i] for i in nodes]) / A
        k = max(1, min(int(p["k"]), len(nodes)))
        sxy = np.array(sim.pos[sim.sink]) / A
        real = []
        for _ in range(int(p["n_real"])):
            cent, _ = kmeans(pts, k, rng, iters=5)
            order = np.argsort(np.arctan2(cent[:, 1] - sxy[1], cent[:, 0] - sxy[0]))
            real.append(cent[order].ravel())
        gan = CentroidGAN(2 * k, rng)
        gan.train(np.array(real), int(p["gan_iters"]))
        cent = gan.sample(1)[0].reshape(k, 2)
        labels = ((pts[:, None, :] - cent[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
        E = np.array([sim.E[i] for i in nodes])
        dsink = np.hypot(pts[:, 0] - sxy[0], pts[:, 1] - sxy[1])
        heads = []
        for c in range(k):
            idx = np.where(labels == c)[0]
            if len(idx) == 0:
                continue
            dc = np.hypot(pts[idx, 0] - cent[c, 0], pts[idx, 1] - cent[c, 1])
            davg = max(dc.mean(), 1e-9)
            s = (p["wE"] * E[idx] / max(E.max(), 1e-12) - p["wD"] * dsink[idx] / max(dsink.max(), 1e-12)
                 - p["wC"] * dc / davg)
            heads.append(nodes[int(idx[np.argmax(s)])])
        return assign_to_heads(sim, heads, sim.scen["p_assign"])

    def next_hop(self, i, kind, final, k=1):
        if kind == COLLECT:
            return final
        sim, p = self.sim, self.p
        Hi = sim.H[i]
        al, H = sim.alive, sim.H
        cand = [j for j in sim.nbr[i] if al[j] and H[j] < Hi]
        if not cand:
            par = sim.parent[i]
            return par if par >= 0 else None
        N = sim.N
        thr = p["cf_thr"]
        cf = self.cf
        ok = [j for j in cand if cf.get(i * N + j, 0.0) <= thr]
        if ok:
            cand = ok
        tau = self.tau[i]
        pr = sim.lq[i]
        aq = sim.adv_q
        E0 = self.E0
        ws = []
        tot = 0.0
        for j in cand:
            ej = 1.0 if j == sim.sink else max(sim.E[j], 0.0) / E0
            w = ((tau.get(j, 1.0) ** p["aT"]) * (ej ** p["aE"]) * (pr[j] ** p["aB"])
                 * ((1.0 / (1.0 + aq[j])) ** p["aL"]) * ((1.0 / (1.0 + H[j])) ** p["aS"]))
            ws.append(w)
            tot += w
        if tot <= 0:
            return cand[0]
        x = sim.prng.random() * tot
        acc = 0.0
        for j, w in zip(cand, ws):
            acc += w
            if acc >= x:
                return j
        return cand[-1]

    def on_tx_result(self, i, j, frame, ok):
        if ok and frame.kind != COLLECT:
            t = self.tau[i]
            t[j] = t.get(j, 1.0) + self.p["Q"] / (1.0 + self.sim.H[j])
            key = i * self.sim.N + j
            self.succ[key] = self.succ.get(key, 0) + 1

    def control(self, now):
        sim = self.sim
        r = 1.0 - self.p["rho"]
        for t in self.tau:
            for j in t:
                v = t[j] * r
                t[j] = v if v > 1e-3 else 1e-3
        N = sim.N
        tc = sim.t_cycle
        self.cf = {key: c * tc / sim.lq[key // N][key % N] for key, c in self.succ.items()}
        self.succ.clear()
