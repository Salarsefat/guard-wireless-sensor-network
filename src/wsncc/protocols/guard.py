"""GUARD: Guard-zone-aware collection with priced, deadline-bounded aggregation.

Flat collection (no cluster heads).  Four mechanisms, each switchable for the
ablation study:

1. Collision-aware collection tree.  Each node estimates, per neighbour, the
   per-attempt success probability s_ij of a data frame *and* its ACK from its
   own transmission outcomes (exponentially weighted, prior p_ij p_ji from beacon
   link estimates).  Under the protocol interference model s_ij falls with the
   hidden part of the receiver's guard zone, so the metric 1/s_ij steers traffic
   onto links that hidden terminals cannot reach (Lemma 1 / Proposition 1 of the
   paper).  Path costs are the shortest-path distances on 1/s_ij plus an
   energy-balance term; a parent is kept within a hysteresis margin only if it is
   strictly closer to the sink, which keeps the parent graph loop-free.
2. Priced dispatch.  A node holding k < K readings transmits only when k reaches
   the control limit m* = min{m : m(m+1) >= 2 c Lam / h} (renewal-reward optimum,
   Theorem 1), where Lam is the node's arrival rate and c the frame price
   (marginal channel congestion plus energy scarcity).  The waiting time of the
   oldest reading is capped by a fraction beta of its deadline slack (Theorem 2).
3. Recovery.  A random delay U(0, rdel) before each MAC retransmission breaks the
   correlation of hidden-terminal collisions (Proposition 3); after the MAC gives
   up, the readings are re-queued up to `nlr` times (network-layer retransmission).
4. Earliest-deadline-first service; readings already past the deadline go last.
"""
from __future__ import annotations

import heapq
import math

from .. import radio as R
from ..sim import TENQ, TGEN
from .base import Protocol


class GUARD(Protocol):
    name = "guard"
    clustered = False
    hdr_extra = 0
    reroute_on_retry = True
    edf = True
    BEACON_EXTRA_BITS = 16      # path cost at 16-bit precision (P_bar for cR > 0 is not charged)
    defaults = {
        # routing
        "route": "meas", "alpha_s": 0.05, "hyst": 0.1, "ebal": 0.5,
        # dispatch
        "batch": True, "h": 0.01, "kappa": 1.0, "u0": 0.0, "cE": 2.0, "cR": 0.0,
        "beta": 0.25, "wmax": 1.0, "tau": 1.0, "dhop": 0.01,
        # recovery and service order
        "nlr": 2, "rdel": 0.01, "edf": True,
        # estimators
        "a": 0.3, "gk_assumed": 1.6,
    }

    def __init__(self, **params):
        super().__init__(**params)
        self.edf = bool(self.p["edf"])

    # ------------------------------------------------------------------ setup
    def init_state(self) -> None:
        sim = self.sim
        p = self.p
        N = sim.N
        sim.edf = self.edf
        sim.edf_tau = float(p["tau"])
        self.tau = float(p["tau"])
        self.lam_hat = [sim.lam] * N           # readings/s entering each node
        self.util = [0.0] * N                  # sensed channel-busy fraction
        self.pw = [0.0] * N                    # power draw [W]
        self.pw_mean = 1e-12
        self.nu = [0.0] * N                    # frames/s on air (model prior only)
        self.rho = [0.0] * N                   # airtime fraction (model prior only)
        self._last = [[0, 0, 0, 0.0, 0.0, 0.0] for _ in range(N)]
        self.s_hat: dict[int, float] = {}
        self.att: dict[int, int] = {}
        self.A_full = R.airtime(R.frame_bits(sim.pack, self.hdr_extra))
        self.parent = list(sim.parent)
        self.D = list(sim.H)
        self.n_route = 0
        if p["route"] == "hybrid":
            self._build_hidden_sets()
        self._route()

    def _build_hidden_sets(self) -> None:
        """G[s][r]: nodes that can corrupt s->r under the protocol model with the
        *assumed* guard factor but lie outside s's carrier-sense range."""
        sim = self.sim
        pos = sim.pos
        gk = float(self.p["gk_assumed"])
        cs = [set(sim.intf[i]) | {i} for i in range(sim.N)]
        self.G = [dict() for _ in range(sim.N)]
        for s in range(sim.n):
            for r in sim.nbr[s]:
                lim = gk * sim.dist[s][r]
                xr, yr = pos[r]
                self.G[s][r] = [t for t in range(sim.n) if t != r and t not in cs[s]
                                and math.hypot(pos[t][0] - xr, pos[t][1] - yr) < lim]

    # --------------------------------------------------------------- routing
    def _prior(self, s: int, r: int) -> float:
        q = self.sim.lq[s][r]
        if self.p["route"] != "hybrid":
            return q
        L = 0.0
        nu, rho, al, A = self.nu, self.rho, self.sim.alive, self.A_full
        for t in self.G[s][r]:
            if al[t]:
                L += nu[t] * A + rho[t]
        return q * math.exp(-L)

    def _link_cost(self, s: int, r: int) -> float:
        sim = self.sim
        if self.p["route"] == "etx":
            succ = sim.lq[s][r]
        else:
            succ = self.s_hat.get(s * sim.N + r)
            if succ is None:
                succ = self._prior(s, r)
        cost = 1.0 / max(succ, 1e-6)
        eb = self.p["ebal"]
        if eb > 0 and r != sim.sink:
            cost += eb * max(sim.scen["E0"] / max(sim.E[r], 1e-9) - 1.0, 0.0)
        return cost

    def _route(self) -> None:
        sim = self.sim
        N = sim.N
        al = sim.alive
        INF = math.inf
        D = [INF] * N
        best = [-1] * N
        s0 = sim.sink
        D[s0] = 0.0
        pq = [(0.0, s0)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > D[u]:
                continue
            for v in sim.nbr[u]:
                if v == s0 or not al[v]:
                    continue
                nd = d + self._link_cost(v, u)
                if nd < D[v] - 1e-12:
                    D[v] = nd
                    best[v] = u
                    heapq.heappush(pq, (nd, v))
        hy = self.p["hyst"]
        par = list(best)
        for v in range(sim.n):
            old = self.parent[v]
            if not al[v] or old is None or old < 0 or old == best[v] or not al[old]:
                continue
            if old not in sim.nbr[v]:
                continue
            # keep the old parent only if strictly closer to the sink (loop freedom)
            if D[old] < D[v] and D[old] + self._link_cost(v, old) <= (1.0 + hy) * D[v]:
                par[v] = old
        self.parent = par
        self.D = D
        self.n_route += 1

    # ------------------------------------------------------------- per second
    def control(self, now: float) -> None:
        sim = self.sim
        a = self.p["a"]
        en = sim.energy
        bits = self.BEACON_EXTRA_BITS
        e_tx, e_rx = en.tx(bits, sim.top.R), en.rx(bits)
        tot_pw, n_al = 0.0, 0
        for i in range(sim.n):
            if not sim.alive[i]:
                continue
            L = self._last[i]
            din = sim.cnt_in[i] - L[0]
            dad = sim.cnt_adm[i] - L[1]
            dair = sim.cnt_air[i] - L[2]
            dairs = sim.air_s[i] - L[3]
            busy = sim.busy_acc[i] - L[4]
            du = sim.used[i] - L[5]
            L[0], L[1], L[2], L[3] = sim.cnt_in[i], sim.cnt_adm[i], sim.cnt_air[i], sim.air_s[i]
            L[4], L[5] = sim.busy_acc[i], sim.used[i]
            self.lam_hat[i] = (1 - a) * self.lam_hat[i] + a * (din + dad)
            self.util[i] = (1 - a) * self.util[i] + a * min(busy, 1.0)
            self.pw[i] = (1 - a) * self.pw[i] + a * du
            self.nu[i] = (1 - a) * self.nu[i] + a * dair
            self.rho[i] = (1 - a) * self.rho[i] + a * dairs
            tot_pw += self.pw[i]
            n_al += 1
            # extra beacon fields (charged like the common beacon)
            k = sum(1 for j in sim.nbr[i] if sim.alive[j])
            sim.charge(i, e_tx + k * e_rx)
            sim.ctrl_bytes += bits // 8
        self.pw_mean = tot_pw / n_al if n_al else 1e-12
        self._route()

    def on_death(self, i: int) -> None:
        super().on_death(i)
        self._route()

    def on_tx_result(self, i, j, frame, ok):
        if self.p["route"] == "etx":
            return
        key = i * self.sim.N + j
        sh = self.s_hat
        prev = sh.get(key)
        if prev is None:
            prev = self._prior(i, j)
        al = self.p["alpha_s"]
        sh[key] = (1.0 - al) * prev + (al if ok else 0.0)

    # ------------------------------------------------------------- decisions
    def frame_price(self, i: int) -> float:
        """Price of one frame in seconds of channel time (marginal congestion of the
        sensed medium plus energy scarcity of the node)."""
        sim = self.sim
        p = self.p
        u = min(self.util[i], 0.95)
        cong = max(u - p["u0"], 0.0) / (1.0 - u)
        scar = max(sim.scen["E0"] / max(sim.E[i], 1e-9) - 1.0, 0.0)
        drain = max(self.pw[i] / max(self.pw_mean, 1e-12) - 1.0, 0.0)
        return self.A_full * (p["kappa"] * cong + p["cE"] * scar + p["cR"] * drain)

    def control_limit(self, i: int) -> int:
        """m* = min{m in 1..K : m(m+1) >= 2 c Lam / h}  (Theorem 1)."""
        x = 2.0 * self.frame_price(i) * self.lam_hat[i] / self.p["h"]
        m = 1
        K = self.sim.pack
        while m < K and m * (m + 1) < x:
            m += 1
        return m

    def next_hop(self, i, kind, final, k=1):
        sim = self.sim
        j = self.parent[i]
        if j is None or j < 0 or not sim.alive[j]:
            j = sim.parent[i]
            if j < 0 or not sim.alive[j]:
                return None
        if not self.p["batch"] or k >= sim.pack or sim.macf[i] is not None:
            return j
        m = self.control_limit(i)
        if k >= m:
            return j
        dq = sim.up[i]
        if not dq:
            return j
        now = sim.now
        oldest = min(rd[TGEN] for rd in dq)
        t_enq = min(rd[TENQ] for rd in dq)
        slack = self.tau - (now - oldest) - self.p["dhop"] * self.D[i]
        budget = min(self.p["beta"] * max(slack, 0.0), (m - k) / max(self.lam_hat[i], 1e-3),
                     self.p["wmax"])
        waited = now - t_enq
        if waited >= budget:
            return j
        self.hold = max(budget - waited, 1e-4)
        return None

    def retry_delay(self, i, frame):
        b = self.p["rdel"]
        return self.sim.prng.random() * b if b > 0 else 0.0

    def on_mac_fail(self, i, frame, reason):
        nl = int(self.p["nlr"])
        if nl <= 0:
            return False
        att = self.att
        keep = True
        for rd in frame.readings:
            a = att.get(rd[0], 0) + 1
            att[rd[0]] = a
            if a > nl:
                keep = False
        if not keep:
            return False
        self.hold = self.sim.prng.random() * max(self.p["rdel"], 0.005) + 1e-4
        return True
