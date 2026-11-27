"""TwinLoop: two-timescale Lyapunov congestion control with collision-priced
batching and facility-location re-clustering (see the paper, Section 4).

Fast loop (every node, every transmission opportunity, O(|neighbours|)):
  * admission:  a routine reading is admitted iff Q_i < V_a (critical readings
                are always admitted and served first);
  * forwarding: per-reading weight towards neighbour j (excess-ETX form)
                    w_ij = (Q_i - Q_j) + theta_0 - theta * (ETX_ij + H_j - H_i) - zeta * Z_j
                a frame of k readings is sent to j* = argmax_j [k w_ij - (V_f + eta Y_j) ETX_ij]
                only if that value is positive; otherwise the node holds (bounded by tau_max);
                a member sends a COLLECT frame to its head h iff
                    k (Q_i - Q_h + theta_c) > (V_f + eta Y_h) ETX_ih;
  * Y_j is the airtime virtual queue of receiver j, updated every second with
    the EWMA fraction phi_j of frames addressed to j that were corrupted by
    interference:  Y_j <- min(max(Y_j + phi_j - phi*, 0), Y_max);
  * Z_j is an optional energy virtual queue (zeta = 0 in all reported experiments).

Slow loop (sink, every T_c seconds, after a node death, or when a head stays congested):
  single-source capacitated facility location (clustering.capacitated_milp)
  with assignment cost lambda_i (a_e e_ij + a_u H_j / K + a_p pi_j), congestion
  price pi_j = qbar_j / B + u_j + phi_j (a_p = 0 in the tuned configuration),
  capacities C_j = kappa (1 - u_j) / t_collect and energy-dependent opening costs.
"""
from __future__ import annotations

import math
import time

from ..clustering import (alive_sensors, capacitated_milp, greedy_capacitated,
                          leachc_heads)
from ..radio import frame_bits, service_cycle
from ..sim import COLLECT, HEAD, TENQ, UPSTREAM
from .base import Protocol


class TwinLoop(Protocol):
    name = "twinloop"
    priority = True
    hdr_extra = 2      # 1 byte backlog + 1 byte quantised airtime price (Z, Y also in beacons)
    defaults = {
        "Va": 48.0, "theta": 16.0, "theta_c": 8.0, "Ve": 1.0, "zeta": 0.0, "zmax": 10.0,
        "Vf": 0.5, "eta": 10.0, "phi_target": 0.15, "ymax": 1.0, "tau_max": 0.5, "theta0": 2.0,
        "reroute": False,
        "T_mission": 3600.0, "Tc": 30.0, "kappa": 0.6, "f0": 3.0, "a_e": 1.0, "a_u": 1.0,
        "a_p": 2.0, "b_E": 1.0, "keep": 0.2, "trig": 0.75, "min_gap": 5.0, "ewma": 0.3,
        "lifo": False, "hold": 0.005, "time_limit": 2.0, "big_m": 1000.0,
        "solver": "milp", "admission": True, "price": True, "capacity": True,
        "recluster": True,
    }

    def init_state(self):
        sim = self.sim
        n, N = sim.n, sim.N
        self.z = [0.0] * N
        self.last_used = [0.0] * N
        self.lam_hat = [max(sim.lam, 0.01)] * n
        self.qbar = [0.0] * N
        self.util = [0.0] * N
        self.last_gen = [0] * n
        self.last_busy = [0.0] * N
        self.last_recluster = -math.inf
        self.solver_log: list[dict] = []
        self.t_collect = service_cycle(frame_bits(1, self.hdr_extra))
        p = self.p
        self.Va = float(p["Va"]) if p["admission"] else math.inf
        self.theta = float(p["theta"])
        self.theta_c = float(p["theta_c"])
        self.Ve = float(p["Ve"])
        self.zeta = float(p["zeta"])
        self.Vf = float(p["Vf"])
        self.eta = float(p["eta"])
        self.tau_max = float(p["tau_max"])
        self.theta0 = float(p["theta0"])
        self.Y = [0.0] * N            # airtime virtual queues (receiver neighbourhoods)
        self.phi = [0.0] * N          # EWMA collision fraction at each receiver
        self.last_att = [0] * N
        self.last_col = [0] * N
        self.reroute_on_retry = bool(p["reroute"])

    # ------------------------------------------------------------ fast loop
    def admit(self, i, cls):
        return cls == 1 or self.sim.qlen[i] < self.Va

    def next_hop(self, i, kind, final, k=1):
        """Drift-plus-penalty decision for a frame that would carry k readings.

        Per-reading weight towards neighbour j (excess-ETX formulation):
            w_ij = (Q_i - Q_j) + theta_0 - theta * (ETX_ij + H_j - H_i) - zeta * Z_j
        where ETX_ij + H_j - H_i >= 0 is the extra expected transmissions of
        routing via j instead of a shortest-ETX next hop (0 on the shortest-path
        tree).  A frame is sent to j* iff
            k * w_ij* - (V_f + eta * Y_j*) * ETX_ij* > 0,
        i.e. its backlog benefit exceeds its expected channel cost.  Y_j is the
        airtime virtual queue of the receiver's neighbourhood, so frames into
        overloaded regions are deferred and batched; readings older than
        tau_max are forwarded regardless (bounded waiting)."""
        sim = self.sim
        Qi = sim.qlen[i]
        aq = sim.adv_q
        Vf, eta, Y = self.Vf, self.eta, self.Y
        if kind == COLLECT:
            etx = 1.0 / sim.lq[i][final]
            if k * (Qi - aq[final] + self.theta_c) - (Vf + eta * Y[final]) * etx > 0:
                return final
            return final if self._age(i, COLLECT) > self.tau_max else None
        Hi = sim.H[i]
        H, al, az, pr = sim.H, sim.alive, sim.adv_z, sim.lq[i]
        th, ze, t0 = self.theta, self.zeta, self.theta0
        best, bw = None, 0.0
        best_any, bw_any = None, -math.inf
        for j in sim.nbr[i]:
            if al[j]:
                etx = 1.0 / pr[j]
                w = Qi - aq[j] + t0 - th * (etx + H[j] - Hi) - ze * az[j]
                ww = k * w - (Vf + eta * Y[j]) * etx
                if ww > bw:
                    bw, best = ww, j
                if H[j] < Hi and w > bw_any:
                    bw_any, best_any = w, j
        if best is None and best_any is not None and self._age(i, UPSTREAM) > self.tau_max:
            return best_any
        return best

    def _age(self, i, kind):
        sim = self.sim
        now = sim.now
        qs = (sim.cpend[i], sim.pend[i]) if kind == COLLECT else (sim.cup[i], sim.up[i])
        a = 0.0
        for dq in qs:
            if dq:
                a = max(a, now - dq[0][TENQ])
        return a

    def control(self, now):
        sim, p = self.sim, self.p
        a = p["ewma"]
        Tm = p["T_mission"]
        zmax = p["zmax"]
        rem = max(Tm - now, 0.1 * Tm)
        pt, ym = p["phi_target"], p["ymax"]
        for i in range(sim.N):
            if not sim.alive[i]:
                continue
            bt = sim.busy_acc[i] + (now - sim.busy_since[i] if sim.busy[i] > 0 else 0.0)
            u = bt - self.last_busy[i]
            self.last_busy[i] = bt
            u = 0.0 if u < 0 else (1.0 if u > 1 else u)
            self.util[i] = (1 - a) * self.util[i] + a * u
            # airtime (collision) virtual queue of the neighbourhood of receiver i:
            # Y <- [Y + phi(t) - phi_target]^+, phi = fraction of frames addressed to i
            # that were corrupted by interference during the last second
            att = sim.cnt_rx_att[i] - self.last_att[i]
            col = sim.cnt_rx_coll[i] - self.last_col[i]
            self.last_att[i], self.last_col[i] = sim.cnt_rx_att[i], sim.cnt_rx_coll[i]
            phi = col / att if att > 0 else 0.0
            self.phi[i] = (1 - a) * self.phi[i] + a * phi
            y = self.Y[i] + phi - pt
            self.Y[i] = 0.0 if y < 0.0 else (ym if y > ym else y)
            if i == sim.sink:
                continue
            spent = sim.used[i] - self.last_used[i]
            self.last_used[i] = sim.used[i]
            budget = max(sim.E[i], 1e-12) / rem
            z = self.z[i] + spent / budget - 1.0
            z = 0.0 if z < 0.0 else (zmax if z > zmax else z)
            self.z[i] = z
            sim.adv_z[i] = z
            g = sim.cnt_gen[i] - self.last_gen[i]
            self.last_gen[i] = sim.cnt_gen[i]
            self.lam_hat[i] = (1 - a) * self.lam_hat[i] + a * g
            self.qbar[i] = (1 - a) * self.qbar[i] + a * sim.qlen[i]
        if p["recluster"] and now - self.last_recluster >= p["min_gap"]:
            thr = p["trig"] * sim.B
            for h in range(sim.n):
                if sim.alive[h] and sim.role[h] == HEAD and self.qbar[h] > thr:
                    self._need_recluster = True
                    break

    # ------------------------------------------------------------ slow loop
    def tick(self, now):
        if not self.p["recluster"]:
            # static-clustering ablation: keep the initial clustering, repair only after deaths
            if self._need_recluster:
                self.recluster()
                self._need_recluster = False
            self.control(now)
            return
        super().tick(now)

    def recluster(self):
        self.last_recluster = self.sim.now
        super().recluster()

    def compute_clusters(self):
        sim, p = self.sim, self.p
        nodes = alive_sensors(sim)
        if not nodes:
            return [], {}
        E0 = sim.scen["E0"]
        Emax = max(sim.E[i] for i in nodes)
        B, pack = sim.B, sim.pack
        en = sim.energy
        bits = sim.full_bits
        e_ref = en.tx(bits, sim.top.R)
        use_price = p["price"]
        lam = {i: max(self.lam_hat[i], 0.05) for i in nodes}
        price = {j: (self.qbar[j] / B + self.util[j] + self.phi[j]) if use_price else 0.0
                 for j in nodes}
        if p["capacity"]:
            cap = {j: p["kappa"] / self.t_collect * max(1.0 - self.util[j], 0.05) for j in nodes}
        else:
            cap = {j: 1e9 for j in nodes}
        Hn = sim.H
        finite = [Hn[j] for j in nodes if Hn[j] < math.inf]
        hmax = max(finite) if finite else 1.0
        up = {j: (Hn[j] if Hn[j] < math.inf else 2 * hmax) / pack for j in nodes}
        f = {}
        for j in nodes:
            fj = p["f0"] * (1.0 + p["b_E"] * (1.0 - sim.E[j] / Emax))
            if sim.role[j] == HEAD and sim.ch[j] == j:
                fj -= p["keep"] * p["f0"]
            if sim.E[j] < 0.05 * E0:
                fj += 100.0
            f[j] = fj
        pa = sim.scen["p_assign"]
        a_e, a_u, a_p = p["a_e"], p["a_u"], p["a_p"]
        pairs = [(j, j, lam[j] * (a_u * up[j] + a_p * price[j])) for j in nodes]
        for i in nodes:
            for j in sim.nbr[i]:
                if j < sim.n and sim.alive[j] and sim.prr[i][j] >= pa:
                    e_hat = (en.tx(bits, sim.dist[i][j]) / e_ref) / sim.lq[i][j]
                    pairs.append((i, j, lam[i] * (a_e * e_hat + a_u * up[j] + a_p * price[j])))
        if p["solver"] == "milp":
            heads, ch_of, info = capacitated_milp(nodes, f, pairs, lam, cap, big_m=p["big_m"],
                                                  time_limit=p["time_limit"])
            if heads is None:
                heads, ch_of, info2 = greedy_capacitated(nodes, f, pairs, lam, cap)
                info["fallback"] = 1
            if p.get("compare_solvers"):   # solver study (E7): same instance, greedy
                gh, gc, ginfo = greedy_capacitated(nodes, f, pairs, lam, cap)
                # evaluate the greedy solution with the MILP objective (incl. overflow)
                load = {j: 0.0 for j in gh}
                obj = sum(f[j] for j in gh)
                cost = {(i, j): c for i, j, c in pairs}
                for i in nodes:
                    j = gc.get(i, i)
                    obj += cost.get((i, j), 0.0)
                    if j in load:
                        load[j] += lam[i]
                over = sum(max(0.0, load[j] - cap[j]) for j in gh)
                info["greedy_obj"] = obj + p["big_m"] * over
                info["greedy_time"] = ginfo["time"]
                info["n_nodes"] = len(nodes)
                info["n_pairs"] = len(pairs)
        elif p["solver"] == "greedy":
            heads, ch_of, info = greedy_capacitated(nodes, f, pairs, lam, cap)
        else:  # "leachc": ablation without the optimisation layer
            heads, ch_of = leachc_heads(sim, 12, sim.np_rng, pa)
            info = {"time": 0.0}
        info["t"] = sim.now
        info["n_heads"] = len(heads)
        self.solver_log.append(info)
        return heads, ch_of
