"""Node placement, link tables and ETX shortest paths."""
from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass, field

from .radio import prr_of_distance


@dataclass
class Topology:
    n: int                                   # sensor nodes 0..n-1, sink = n
    pos: list[tuple[float, float]]
    R: float
    nbr: list[list[int]] = field(default_factory=list)       # routing neighbours
    prr: list[dict[int, float]] = field(default_factory=list)  # PRR of neighbour links
    lq: list[dict[int, float]] = field(default_factory=list)   # p_ij * p_ji (data and ACK)
    dist: list[dict[int, float]] = field(default_factory=list)
    intf: list[list[int]] = field(default_factory=list)      # within carrier-sense range

    @property
    def sink(self) -> int:
        return self.n


def _place(n: int, area: float, kind: str, rng: random.Random) -> list[tuple[float, float]]:
    if kind == "uniform":
        return [(rng.uniform(0, area), rng.uniform(0, area)) for _ in range(n)]
    if kind == "clustered":  # Thomas process: 6 parents, Gaussian offspring
        parents = [(rng.uniform(0.15 * area, 0.85 * area), rng.uniform(0.15 * area, 0.85 * area))
                   for _ in range(6)]
        sigma = 0.14 * area
        pts = []
        for _ in range(n):
            px, py = parents[rng.randrange(len(parents))]
            x = min(max(rng.gauss(px, sigma), 0.0), area)
            y = min(max(rng.gauss(py, sigma), 0.0), area)
            pts.append((x, y))
        return pts
    raise ValueError(f"unknown topology kind {kind!r}")


def _sink_position(area: float, where: str) -> tuple[float, float]:
    if where == "center":
        return (area / 2.0, area / 2.0)
    if where == "corner":
        return (0.0, 0.0)
    raise ValueError(f"unknown sink position {where!r}")


def build_topology(n: int, area: float, R: float, *, kind: str = "uniform",
                   sink: str = "center", intf_factor: float = 1.5,
                   prr_width: float = 0.06, shadow_db: float = 2.0,
                   p_route: float = 0.5, seed: int = 0,
                   max_tries: int = 500) -> Topology:
    """Generate a connected topology (every node reaches the sink over
    links with PRR >= p_route).  Deterministic for a given seed."""
    rng = random.Random(seed)
    eta = 3.0  # path-loss exponent used to convert dB shadowing to distance
    for _ in range(max_tries):
        pos = _place(n, area, kind, rng) + [_sink_position(area, sink)]
        N = n + 1
        nbr: list[list[int]] = [[] for _ in range(N)]
        prr: list[dict[int, float]] = [dict() for _ in range(N)]
        dist: list[dict[int, float]] = [dict() for _ in range(N)]
        intf: list[list[int]] = [[] for _ in range(N)]
        RI = intf_factor * R
        dmax_link = 1.6 * R
        for i in range(N):
            xi, yi = pos[i]
            for j in range(i + 1, N):
                dx = xi - pos[j][0]
                dy = yi - pos[j][1]
                d = math.hypot(dx, dy)
                if d <= RI:
                    intf[i].append(j)
                    intf[j].append(i)
                if d <= dmax_link:
                    shadow = rng.gauss(0.0, shadow_db)
                    d_eff = d * 10 ** (shadow / (10.0 * eta))
                    p = prr_of_distance(d_eff, R, prr_width)
                    if p >= p_route:
                        nbr[i].append(j)
                        nbr[j].append(i)
                        prr[i][j] = p
                        prr[j][i] = p
                        dist[i][j] = d
                        dist[j][i] = d
        # every routing link must also be in the interference/carrier-sense set
        for i in range(N):
            s = set(intf[i])
            for j in nbr[i]:
                if j not in s:
                    intf[i].append(j)
                    s.add(j)
        # connectivity check (BFS from sink)
        seen = {n}
        stack = [n]
        while stack:
            u = stack.pop()
            for v in nbr[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        if len(seen) == N:
            # per-attempt link-layer success needs both the data frame (p_ij) and
            # the ACK (p_ji); ETX = 1/(p_ij p_ji) as defined by De Couto et al.
            lq = [{j: prr[i][j] * prr[j][i] for j in nbr[i]} for i in range(N)]
            return Topology(n=n, pos=pos, R=R, nbr=nbr, prr=prr, lq=lq, dist=dist,
                            intf=intf)
    raise RuntimeError("could not generate a connected topology")


def etx_distances(top: Topology, alive: list[bool]) -> tuple[list[float], list[int]]:
    """Dijkstra from the sink with link cost ETX = 1/(p_ij p_ji) over alive nodes.

    Returns (H, parent) where H[i] is the ETX cost-to-sink and parent[i] the
    next hop on the minimum-ETX path (-1 if unreachable)."""
    N = top.n + 1
    INF = float("inf")
    H = [INF] * N
    parent = [-1] * N
    s = top.sink
    H[s] = 0.0
    pq = [(0.0, s)]
    while pq:
        h, u = heapq.heappop(pq)
        if h > H[u]:
            continue
        for v in top.nbr[u]:
            if not alive[v]:
                continue
            nh = h + 1.0 / top.lq[u][v]
            if nh < H[v] - 1e-12:
                H[v] = nh
                parent[v] = u
                heapq.heappush(pq, (nh, v))
    return H, parent
