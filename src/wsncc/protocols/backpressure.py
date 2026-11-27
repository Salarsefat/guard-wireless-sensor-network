"""Backpressure baselines.

BCP   Backpressure Collection Protocol (flat): next hop maximises
      (Q_i - Q_j - V * ETX_ij) * q_ij, transmit only if positive, LIFO queue.
CBPR  cluster-based backpressure: CBPR head rule (highest energy, shortest
      distance to the sink in each partition), forwarding to the neighbour closer
      to the sink that maximises (Q_i - Q_j) * q_ij (if positive), FIFO.

q_ij = p_ij p_ji is the per-attempt success probability (data and ACK).
"""
from __future__ import annotations

from ..clustering import cbpr_heads
from ..sim import COLLECT
from .base import Protocol


class BCP(Protocol):
    name = "bcp"
    clustered = False
    lifo = True
    hdr_extra = 2
    defaults = {"V": 4.0, "hold": 0.01}   # V=4: smallest loop-free value in pilot runs

    def next_hop(self, i, kind, final, k=1):
        sim = self.sim
        Qi = sim.qlen[i]
        V = self.p["V"]
        al, aq, pr = sim.alive, sim.adv_q, sim.lq[i]
        best, bw = None, 0.0
        for j in sim.nbr[i]:
            if al[j]:
                # BCP weight: (backlog differential - V * ETX) x link-rate estimate
                w = (Qi - aq[j] - V / pr[j]) * pr[j]
                if w > bw:
                    bw, best = w, j
        return best


class CBPR(Protocol):
    name = "cbpr"
    hdr_extra = 2
    defaults = {"k": 12, "Tc": 30.0, "hold": 0.01}

    def compute_clusters(self):
        sim = self.sim
        return cbpr_heads(sim, int(self.p["k"]), sim.np_rng, sim.scen["p_assign"])

    def next_hop(self, i, kind, final, k=1):
        if kind == COLLECT:
            return final
        sim = self.sim
        Qi = sim.qlen[i]
        Hi = sim.H[i]
        al, aq, H, pr = sim.alive, sim.adv_q, sim.H, sim.lq[i]
        best, bw = None, 0.0
        for j in sim.nbr[i]:
            if al[j] and H[j] < Hi:
                w = (Qi - aq[j]) * pr[j]      # queue differential x link quality
                if w > bw:
                    bw, best = w, j
        return best
