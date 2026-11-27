"""Development candidate for the redesigned method (flat collection).

Components (each switchable for ablation):
  route : "etx"  static bidirectional-ETX tree
          "ht"   hidden-terminal-aware ETX: link cost 1/(q (1 - c_hat)) where c_hat is
                 a Poisson-overlap estimate of the collision probability caused by
                 transmitters inside the receiver's protocol-model guard zone that the
                 sender cannot carrier-sense
  batch : "none" send whatever is queued as soon as the MAC is free
          "sqrt" congestion-priced dispatch threshold m* = min{m : m(m+1) >= 2 c Lam / h}
                 with a slack-bounded waiting budget
  nlr   : network-layer retransmissions after the MAC gives up (0 = off)
  edf   : earliest-deadline-first service order
  ebal  : residual-energy weight in the parent cost
"""
from __future__ import annotations

import math

from .. import radio as R
from ..sim import TGEN, TENQ
from .base import Protocol


class V2(Protocol):
    name = "v2"
    clustered = False
    hdr_extra = 2
    reroute_on_retry = True
    defaults = {
        "route": "ht", "batch": "sqrt", "nlr": 2, "rb": 0.02, "edf": True,
        "tau": 1.0, "beta": 0.25, "h": 0.01, "kappa": 1.0, "mu": 0.0,
        "hyst": 0.1, "ebal": 0.0, "a": 0.3, "dhop": 0.01, "tick_route": 1.0,
        "c0": 1.0, "wmax": 1.0, "hdr": 2, "gk_assumed": None, "rdel": 0.0, "alpha_s": 0.05, "u0": 0.0, "cE": 0.0, "cR": 0.0,
    }

    def __init__(self, **params):
        super().__init__(**params)
        self.hdr_extra = int(self.p["hdr"])

    # ------------------------------------------------------------------ setup
    def init_state(self) -> None:
        sim = self.sim
        p = self.p
        N = sim.N
        self.edf = bool(p["edf"])
        self.edf_tau = float(p["tau"])
        sim.edf = self.edf
        sim.edf_tau = self.edf_tau
        self.tau = float(p["tau"])
        self.lam_hat = [sim.lam] * N           # readings/s entering each node (EWMA)
        self.nu = [0.0] * N                    # frames/s put on air (EWMA)
        self.rho = [0.0] * N                   # airtime fraction (EWMA)
        self.util = [0.0] * N                  # sensed busy fraction (EWMA)
        self.pw = [0.0] * N                    # power draw (EWMA, W)
        self.pw_mean = 1e-12                   # network-mean power draw (disseminated)
        self._last_used = [0.0] * N
        self._last_in = [0] * N
        self._last_adm = [0] * N
        self._last_air = [0] * N
        self._last_airs = [0.0] * N
        self._last_busy = [0.0] * N
        self.att: dict[int, int] = {}          # network-layer attempts per reading id
        self.A_full = R.airtime(R.frame_bits(sim.pack, self.hdr_extra))
        self.A1 = R.airtime(R.frame_bits(1, self.hdr_extra))
        self.parent = list(sim.parent)
        self.D = list(sim.H)
        self.s_hat: dict[int, float] = {}       # measured per-attempt success (EWMA)
        self._att: dict[int, list] = {}
        self._build_hidden_sets()
        self._route()

    def _build_hidden_sets(self) -> None:
        """G[s][r]: nodes t that can corrupt s->r under the protocol model
        (d(t, r) < (1 + Delta) d(s, r)) but are outside s's carrier-sense set."""
        sim = self.sim
        pos = sim.pos
        gk = sim.gk
        N = sim.N
        if self.p["gk_assumed"] is not None:
            gk = float(self.p["gk_assumed"])
        cs = [set(sim.intf[i]) | {i} for i in range(N)]
        self.cs = cs
        self.G = [dict() for _ in range(N)]
        for s in range(sim.n):
            xs, ys = pos[s]
            for r in sim.nbr[s]:
                lim = gk * sim.dist[s][r]
                xr, yr = pos[r]
                hid = []
                for t in range(sim.n):
                    if t == r or t in cs[s]:
                        continue
                    if math.hypot(pos[t][0] - xr, pos[t][1] - yr) < lim:
                        hid.append(t)
                self.G[s][r] = hid

    # --------------------------------------------------------------- routing
    def _link_cost(self, s: int, r: int) -> float:
        sim = self.sim
        q = sim.lq[s][r]
        rt = self.p["route"]
        if rt == "meas" or rt == "hybrid":
            succ = self.s_hat.get(s * sim.N + r)
            if succ is None:
                succ = q if rt == "meas" else q * (1.0 - self._c_model(s, r))
            return 1.0 / max(succ, 1e-6) + self._extra(s, r, succ)
        c = 0.0 if rt == "etx" else self._c_model(s, r)
        succ = q * (1.0 - c)
        return 1.0 / max(succ, 1e-6) + self._extra(s, r, succ)

    def _c_model(self, s: int, r: int) -> float:
        """Poisson-overlap collision probability of s->r from hidden transmitters."""
        sim = self.sim
        A = self.A_full
        nu, rho, al = self.nu, self.rho, sim.alive
        L = 0.0
        g = self.G[s][r]
        for t in g:
            if al[t]:
                L += nu[t] * A + rho[t]
        cs = self.cs[s]
        for t in self.children[r]:          # hidden children of r keep r busy
            if t != s and t not in cs and t not in g:
                L += nu[t] * A + rho[t]
        return 1.0 - math.exp(-L)

    def _extra(self, s: int, r: int, succ: float) -> float:
        sim = self.sim
        cost = 0.0
        mu = self.p["mu"]
        if mu > 0:
            cost += mu * (1.0 - succ) ** (R.MAC_MAX_FRAME_RETRIES + 1)
        eb = self.p["ebal"]
        if eb > 0 and r != sim.sink:
            E0 = sim.scen["E0"]
            cost += eb * (E0 / max(sim.E[r], 1e-9) - 1.0)
        return cost

    def _route(self) -> None:
        """Dijkstra from the sink on the current link costs, then keep the old
        parent when it is within the hysteresis margin and strictly closer to the
        sink (D[parent] < D[node] guarantees a loop-free parent graph)."""
        import heapq
        sim = self.sim
        N = sim.N
        al = sim.alive
        self.children = [[] for _ in range(N)]
        for i in range(sim.n):
            pj = self.parent[i]
            if al[i] and pj is not None and pj >= 0:
                self.children[pj].append(i)
        INF = math.inf
        D = [INF] * N
        best = [-1] * N
        cost_of = {}
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
                c = self._link_cost(v, u)
                cost_of[(v, u)] = c
                nd = d + c
                if nd < D[v] - 1e-12:
                    D[v] = nd
                    best[v] = u
                    heapq.heappush(pq, (nd, v))
        hy = self.p["hyst"]
        new_parent = list(best)
        for v in range(sim.n):
            old = self.parent[v]
            if not al[v] or old is None or old < 0 or old == best[v] or not al[old]:
                continue
            c_old = cost_of.get((v, old))
            if c_old is None:
                c_old = self._link_cost(v, old)
            if D[old] < D[v] and D[old] + c_old <= (1.0 + hy) * D[v]:
                new_parent[v] = old
        self.parent = new_parent
        self.D = D

    def on_tx_result(self, i, j, frame, ok):
        if self.p["route"] in ("meas", "hybrid"):
            key = i * self.sim.N + j
            sh = self.s_hat
            prev = sh.get(key)
            if prev is None:
                q = self.sim.lq[i][j]
                prev = q if self.p["route"] == "meas" else q * (1.0 - self._c_model(i, j))
            al = self.p["alpha_s"]
            sh[key] = (1.0 - al) * prev + al * (1.0 if ok else 0.0)

    # ------------------------------------------------------------- per second
    def control(self, now: float) -> None:
        sim = self.sim
        a = self.p["a"]

        for i in range(sim.n):
            if not sim.alive[i]:
                continue
            din = sim.cnt_in[i] - self._last_in[i]
            dad = sim.cnt_adm[i] - self._last_adm[i]
            dair = sim.cnt_air[i] - self._last_air[i]
            dairs = sim.air_s[i] - self._last_airs[i]
            busy = sim.busy_acc[i] - self._last_busy[i]
            self._last_in[i] = sim.cnt_in[i]
            self._last_adm[i] = sim.cnt_adm[i]
            self._last_air[i] = sim.cnt_air[i]
            self._last_airs[i] = sim.air_s[i]
            self._last_busy[i] = sim.busy_acc[i]
            self.lam_hat[i] = (1 - a) * self.lam_hat[i] + a * (din + dad)
            self.nu[i] = (1 - a) * self.nu[i] + a * dair
            self.rho[i] = (1 - a) * self.rho[i] + a * dairs
            self.util[i] = (1 - a) * self.util[i] + a * min(busy, 1.0)
            du = sim.used[i] - self._last_used[i]
            self._last_used[i] = sim.used[i]
            self.pw[i] = (1 - a) * self.pw[i] + a * du
        alive = [self.pw[i] for i in range(sim.n) if sim.alive[i]]
        self.pw_mean = sum(alive) / len(alive) if alive else 1e-12
        if self.p["route"] != "etx" or self.p["ebal"] > 0:
            self._route()
        elif sim.parent != self.parent:
            self.parent = list(sim.parent)
            self.D = list(sim.H)

    def on_death(self, i: int) -> None:
        super().on_death(i)
        self._route()

    # ------------------------------------------------------------- decisions
    def _threshold(self, i: int) -> int:
        """Congestion-priced dispatch threshold (renewal-reward optimum)."""
        u = min(self.util[i], 0.95)
        cong = max(u - self.p["u0"], 0.0) / (1.0 - u)
        sim = self.sim
        scar = max(sim.scen["E0"] / max(sim.E[i], 1e-9) - 1.0, 0.0)
        drain = max(self.pw[i] / max(self.pw_mean, 1e-12) - 1.0, 0.0)
        c = self.A_full * (self.p["c0"] + self.p["kappa"] * cong + self.p["cE"] * scar
                           + self.p["cR"] * drain)
        x = 2.0 * c * self.lam_hat[i] / self.p["h"]
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
        if self.p["batch"] == "none" or k >= sim.pack:
            return j
        if sim.macf[i] is not None:           # a retransmission: never hold
            return j
        m = self._threshold(i)
        if k >= m:
            return j
        # slack-bounded waiting budget for the oldest queued reading
        dq = sim.up[i]
        if not dq:
            return j
        now = sim.now
        oldest = min(rd[TGEN] for rd in dq)
        t_enq = min(rd[TENQ] for rd in dq)
        slack = self.tau - (now - oldest) - self.p["dhop"] * self.D[i]
        budget = self.p["beta"] * max(slack, 0.0)
        lam = max(self.lam_hat[i], 1e-3)
        budget = min(budget, (m - k) / lam, self.p["wmax"])  # never beyond the expected fill time
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
        self.hold = self.sim.prng.random() * self.p["rb"] + 1e-4
        return True
