"""C-ETX substrate and the three rate-control baselines of the reference article.

C-ETX      LEACH-C-like centralised clustering + minimum-ETX routing, FIFO, no
           congestion control.  It is the common substrate on which the
           ACSRO-, FSFG- and DRL-CaCC-inspired controllers are evaluated, so
           that those baselines differ only in their congestion-control logic.
ACSRO-like adaptive cuckoo-search allocation of member sending rates so that the
           arrival rate at each cluster head stays within its service rate.
FSFG-like  zero-order Sugeno fuzzy inference of a congestion degree from buffer
           occupancy, arrival/service ratio and channel contention; membership
           breakpoints are tuned with a genetic algorithm (offline, on the
           validation seeds only).
RL-CC      tabular Q-learning rate controller inspired by DRL-CaCC; the Q-table
           is trained offline on dedicated training seeds and frozen at test.
"""
from __future__ import annotations

import math

import numpy as np

from ..sim import COLLECT, HEAD
from .base import Protocol


class CETX(Protocol):
    name = "cetx"
    defaults = {"k": 12, "Tc": 30.0}

    def next_hop(self, i, kind, final, k=1):
        if kind == COLLECT:
            return final
        p = self.sim.parent[i]
        return p if p >= 0 else None


class _RateControlled(CETX):
    """C-ETX plus per-source admission with a rate limit or admission probability."""

    def init_state(self):
        sim = self.sim
        n = sim.n
        self.phi = [1.0] * n                 # admission probability
        self.next_ctrl = float(self.p.get("Tctrl", 2.0))
        self.last_gen = [0] * n
        self.last_out = [0] * n
        self.last_in = [0] * n
        self.last_inc = [0] * n
        self.last_cca = [0] * n
        self.last_ccaf = [0] * n
        self.last_ok = [0] * n
        self.last_fail = [0] * n
        self.d_hat = [sim.lam] * n

    def admit(self, i, cls):
        ph = self.phi[i]
        return ph >= 1.0 or self.sim.prng.random() < ph

    def _rates(self, T):
        """Per-node measured rates over the last control period:
        generated g, admitted a, forwarded o, received inn, received-in-COLLECT c."""
        sim = self.sim
        n = sim.n
        g = [0.0] * n
        a = [0.0] * n
        o = [0.0] * n
        inn = [0.0] * n
        c = [0.0] * n
        if not hasattr(self, "last_adm"):
            self.last_adm = [0] * n
        for i in range(n):
            g[i] = (sim.cnt_gen[i] - self.last_gen[i]) / T
            a[i] = (sim.cnt_adm[i] - self.last_adm[i]) / T
            o[i] = (sim.cnt_out[i] - self.last_out[i]) / T
            inn[i] = (sim.cnt_in[i] - self.last_in[i]) / T
            c[i] = (sim.cnt_in_collect[i] - self.last_inc[i]) / T
            self.last_gen[i] = sim.cnt_gen[i]
            self.last_adm[i] = sim.cnt_adm[i]
            self.last_out[i] = sim.cnt_out[i]
            self.last_in[i] = sim.cnt_in[i]
            self.last_inc[i] = sim.cnt_in_collect[i]
            self.d_hat[i] = 0.7 * self.d_hat[i] + 0.3 * g[i]
        self._adm_rate = a
        self._collect_rate = c
        return g, o, inn


# ----------------------------------------------------------------- ACSRO-like
def _levy(rng, size, beta=1.5):
    num = math.gamma(1 + beta) * math.sin(math.pi * beta / 2)
    den = math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)
    sigma = (num / den) ** (1 / beta)
    u = rng.normal(0, sigma, size)
    v = rng.normal(0, 1, size)
    return u / np.abs(v) ** (1 / beta)


class ACSRO(_RateControlled):
    name = "acsro"
    defaults = {"k": 12, "Tc": 30.0, "Tctrl": 2.0, "nests": 10, "iters": 20, "pa": 0.25,
                "w_l": 4.0, "w_f": 0.5, "q_low": 0.25}

    def control(self, now):
        if now < self.next_ctrl:
            return
        p = self.p
        T = float(p["Tctrl"])
        self.next_ctrl = now + T
        sim = self.sim
        g, o, inn = self._rates(T)
        rng = sim.np_rng
        members: dict[int, list[int]] = {}
        for i in range(sim.n):
            if sim.alive[i]:
                members.setdefault(sim.ch[i], []).append(i)
        for h, mem in members.items():
            if not sim.alive[h]:
                continue
            q = sim.nq[h] / sim.B
            d = np.array([max(self.d_hat[i], 1e-3) for i in mem])
            if q < p["q_low"]:
                for i in mem:   # uncongested parent: relax limits multiplicatively
                    self.phi[i] = min(1.0, self.phi[i] * 1.25 + 0.05)
                continue
            # service rate available to members = parent output minus transit arrivals
            transit = max(inn[h] - self._collect_rate[h], 0.0)
            mu = max(o[h] * (1.0 - q) - transit, 0.05 * d.sum())
            r = self._cuckoo(d, mu, rng)
            for idx, i in enumerate(mem):
                self.phi[i] = float(min(1.0, max(0.02, r[idx] / d[idx])))

    def _cuckoo(self, d, mu, rng):
        p = self.p
        nn, it, pa = int(p["nests"]), int(p["iters"]), float(p["pa"])
        lo, hi = 0.02 * d, d
        dim = len(d)

        def fit(x):  # maximise delivered rate, penalise overload and unfairness
            tot = x.sum(axis=-1)
            over = np.maximum(tot - mu, 0.0)
            fair = np.std(x / d, axis=-1)
            return tot - p["w_l"] * over * dim - p["w_f"] * fair * d.sum()

        nests = lo + (hi - lo) * rng.random((nn, dim))
        f = fit(nests)
        for t in range(it):
            alpha = 0.1 * (1.0 - t / it) + 0.01           # adaptive step size
            best = nests[np.argmax(f)]
            new = nests + alpha * _levy(rng, (nn, dim)) * (nests - best) * (hi - lo)
            new = np.clip(new, lo, hi)
            fn = fit(new)
            better = fn > f
            nests[better], f[better] = new[better], fn[better]
            # abandon a fraction pa of the worst nests
            k = max(1, int(pa * nn))
            worst = np.argsort(f)[:k]
            nests[worst] = lo + (hi - lo) * rng.random((k, dim))
            f[worst] = fit(nests[worst])
        return nests[np.argmax(f)]


# ----------------------------------------------------------------- FSFG-like
def _tri3(x, c):
    """Memberships (low, med, high) of x in [0,1] for triangles peaking at 0, c, 1."""
    x = min(max(x, 0.0), 1.0)
    low = max(0.0, 1.0 - x / c) if c > 0 else 0.0
    high = max(0.0, (x - c) / (1.0 - c)) if c < 1 else 0.0
    med = 1.0 - low - high
    return (low, max(med, 0.0), high)


class FSFG(_RateControlled):
    name = "fsfg"
    defaults = {"k": 12, "Tc": 30.0, "Tctrl": 2.0, "c_buf": 0.5, "c_cf": 0.5, "c_cca": 0.5,
                "gain": 1.0}
    _LV = (0.0, 0.5, 1.0)

    def control(self, now):
        if now < self.next_ctrl:
            return
        p = self.p
        T = float(p["Tctrl"])
        self.next_ctrl = now + T
        sim = self.sim
        g, o, inn = self._rates(T)
        n = sim.n
        cdeg = [0.0] * n
        lv = self._LV
        for i in range(n):
            if not sim.alive[i]:
                continue
            b = sim.nq[i] / sim.B
            # unserved fraction of the admitted arrivals (0 when the node keeps up)
            arr = self._adm_rate[i] + inn[i]
            cf = min(max((arr - o[i]) / arr, 0.0), 1.0) if arr > 0 else 0.0
            cca = sim.cnt_cca[i] - self.last_cca[i]
            ccaf = sim.cnt_ccafail[i] - self.last_ccaf[i]
            self.last_cca[i], self.last_ccaf[i] = sim.cnt_cca[i], sim.cnt_ccafail[i]
            u = ccaf / cca if cca > 0 else 0.0
            m1, m2, m3 = _tri3(b, p["c_buf"]), _tri3(cf, p["c_cf"]), _tri3(u, p["c_cca"])
            num = den = 0.0
            for a in range(3):
                if m1[a] == 0.0:
                    continue
                for bb in range(3):
                    if m2[bb] == 0.0:
                        continue
                    for c in range(3):
                        w = m1[a] * m2[bb] * m3[c]
                        if w > 0.0:      # zero-order Sugeno consequent: mean level
                            num += w * (lv[a] + lv[bb] + lv[c]) / 3.0
                            den += w
            cdeg[i] = num / den if den > 0 else 0.0
        for i in range(n):
            if sim.alive[i]:
                c = max(cdeg[i], cdeg[sim.ch[i]])
                self.phi[i] = min(1.0, max(0.02, 1.0 - p["gain"] * c))


# --------------------------------------------------------------------- RL-CC
class RLCC(_RateControlled):
    """Tabular Q-learning admission controller (shared table, 48 states x 3 actions)."""

    name = "rlcc"
    defaults = {"k": 12, "Tc": 30.0, "Tctrl": 1.0, "alpha": 0.1, "gamma": 0.9, "eps": 0.1,
                "w_l": 1.0, "w_b": 1.0, "training": False}
    ACTIONS = (0.7, 1.0, 1.15)

    def __init__(self, qtable=None, **params):
        super().__init__(**params)
        self.Q = np.zeros((48, 3)) if qtable is None else np.asarray(qtable, dtype=float)

    def init_state(self):
        super().init_state()
        n = self.sim.n
        self.s_prev = [-1] * n
        self.a_prev = [1] * n

    @staticmethod
    def _bucket(x, edges):
        for k, e in enumerate(edges):
            if x < e:
                return k
        return len(edges)

    def control(self, now):
        if now < self.next_ctrl:
            return
        p = self.p
        T = float(p["Tctrl"])
        self.next_ctrl = now + T
        sim = self.sim
        g, o, inn = self._rates(T)
        rng = sim.prng
        train = bool(p["training"])
        Q = self.Q
        for i in range(sim.n):
            if not sim.alive[i]:
                continue
            ok = sim.cnt_txok[i] - self.last_ok[i]
            fail = sim.cnt_txfail[i] - self.last_fail[i]
            self.last_ok[i], self.last_fail[i] = sim.cnt_txok[i], sim.cnt_txfail[i]
            loss = fail / (ok + fail) if ok + fail > 0 else 0.0
            b = sim.nq[i] / sim.B
            nxt = sim.ch[i] if sim.ch[i] != i else sim.parent[i]
            pb = sim.adv_q[nxt] / sim.B if nxt >= 0 else 0.0
            s = (self._bucket(b, (0.25, 0.5, 0.75)) * 12 + self._bucket(pb, (0.25, 0.5, 0.75)) * 3
                 + self._bucket(loss, (0.05, 0.2)))
            if train and self.s_prev[i] >= 0:
                adm_frac = self.phi[i]
                r = adm_frac - p["w_l"] * loss - p["w_b"] * b
                sp, ap = self.s_prev[i], self.a_prev[i]
                Q[sp, ap] += p["alpha"] * (r + p["gamma"] * Q[s].max() - Q[sp, ap])
            if train and rng.random() < p["eps"]:
                a = rng.randrange(3)
            else:
                a = int(np.argmax(Q[s]))
            self.phi[i] = min(1.0, max(0.02, self.phi[i] * self.ACTIONS[a]))
            self.s_prev[i], self.a_prev[i] = s, a
