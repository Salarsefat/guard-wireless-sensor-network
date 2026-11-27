"""Clustering algorithms used by the protocols.

* k-means partitioning (k-means++ seeding, Lloyd iterations)
* LEACH-C-like head selection (heads among nodes with above-average energy)
* CBPR head rule (highest energy, shortest distance to the sink per partition)
* capacitated, price-coupled facility-location MILP used by TwinLoop
"""
from __future__ import annotations

import math
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


def alive_sensors(sim) -> list[int]:
    return [i for i in range(sim.n) if sim.alive[i]]


def kmeans(points: np.ndarray, k: int, rng: np.random.Generator, iters: int = 12):
    """Plain k-means with k-means++ seeding.  Returns (centroids, labels)."""
    m = len(points)
    k = max(1, min(k, m))
    cent = np.empty((k, points.shape[1]))
    cent[0] = points[rng.integers(m)]
    d2 = np.sum((points - cent[0]) ** 2, axis=1)
    for c in range(1, k):
        s = d2.sum()
        idx = rng.integers(m) if s <= 0 else rng.choice(m, p=d2 / s)
        cent[c] = points[idx]
        d2 = np.minimum(d2, np.sum((points - cent[c]) ** 2, axis=1))
    labels = np.zeros(m, dtype=int)
    for _ in range(iters):
        dist = ((points[:, None, :] - cent[None, :, :]) ** 2).sum(axis=2)
        labels = dist.argmin(axis=1)
        for c in range(k):
            sel = labels == c
            if sel.any():
                cent[c] = points[sel].mean(axis=0)
            else:  # re-seed an empty cluster at the farthest point
                cent[c] = points[dist.min(axis=1).argmax()]
    return cent, labels


def assign_to_heads(sim, heads: list[int], p_assign: float):
    """Members join the nearest head reachable over a link with PRR >= p_assign.
    Nodes that cannot reach any head are promoted greedily to singleton heads
    (and their unassigned neighbours join them)."""
    heads = [h for h in dict.fromkeys(heads) if sim.alive[h]]
    hs = set(heads)
    ch_of: dict[int, int] = {}
    unassigned = []
    for i in alive_sensors(sim):
        if i in hs:
            continue
        best, bd = -1, math.inf
        for j in sim.nbr[i]:
            if j in hs and sim.prr[i][j] >= p_assign:
                d = sim.dist[i][j]
                if d < bd:
                    best, bd = j, d
        if best < 0:
            unassigned.append(i)
        else:
            ch_of[i] = best
    left = set(unassigned)
    for i in unassigned:
        if i not in left:
            continue
        left.discard(i)
        heads.append(i)
        hs.add(i)
        for j in sim.nbr[i]:
            if j in left and sim.prr[i][j] >= p_assign:
                ch_of[j] = i
                left.discard(j)
    return heads, ch_of


def leachc_heads(sim, k: int, rng: np.random.Generator, p_assign: float):
    nodes = alive_sensors(sim)
    if not nodes:
        return [], {}
    pts = np.array([sim.pos[i] for i in nodes])
    cent, _ = kmeans(pts, k, rng)
    E = np.array([sim.E[i] for i in nodes])
    eligible = [i for i, e in zip(nodes, E) if e >= E.mean()]
    if len(eligible) < k:
        eligible = nodes
    epts = np.array([sim.pos[i] for i in eligible])
    heads, used = [], set()
    for c in cent:
        order = np.argsort(((epts - c) ** 2).sum(axis=1))
        for o in order:
            cand = eligible[int(o)]
            if cand not in used:
                heads.append(cand)
                used.add(cand)
                break
    return assign_to_heads(sim, heads, p_assign)


def cbpr_heads(sim, k: int, rng: np.random.Generator, p_assign: float):
    """CBPR rule: per partition, the head maximises E/Emax - d_sink/d_max."""
    nodes = alive_sensors(sim)
    if not nodes:
        return [], {}
    pts = np.array([sim.pos[i] for i in nodes])
    _, labels = kmeans(pts, k, rng)
    sx, sy = sim.pos[sim.sink]
    dsink = np.hypot(pts[:, 0] - sx, pts[:, 1] - sy)
    E = np.array([sim.E[i] for i in nodes])
    score = E / max(E.max(), 1e-12) - dsink / max(dsink.max(), 1e-12)
    heads = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        heads.append(nodes[int(idx[np.argmax(score[idx])])])
    return assign_to_heads(sim, heads, p_assign)


def capacitated_milp(nodes: list[int], open_cost: dict[int, float],
                     pairs: list[tuple[int, int, float]], demand: dict[int, float],
                     capacity: dict[int, float], *, big_m: float = 1e3,
                     time_limit: float = 2.0, mip_gap: float = 1e-3):
    """Single-source capacitated facility location solved with HiGHS.

    min  sum_j f_j y_j + sum_(i,j) c_ij x_ij + M sum_j s_j
    s.t. sum_j x_ij = 1                       for every node i
         x_ij <= y_j                          for every candidate pair, i != j
         x_jj  = y_j                          (an open head serves itself)
         sum_i lam_i x_ij <= C_j y_j + s_j    (soft capacity, slack s_j >= 0)
         x, y binary
    Returns (heads, ch_of, info)."""
    t0 = time.perf_counter()
    m = len(nodes)
    col = {j: k for k, j in enumerate(nodes)}
    P = len(pairs)
    nv = m + P + m
    c = np.zeros(nv)
    for j in nodes:
        c[col[j]] = open_cost[j]
    for p, (i, j, cij) in enumerate(pairs):
        c[m + p] = cij
    c[m + P:] = big_m
    rows, cols, vals = [], [], []
    lb, ub = [], []
    r = 0
    # assignment
    by_i: dict[int, list[int]] = {i: [] for i in nodes}
    by_j: dict[int, list[int]] = {j: [] for j in nodes}
    self_idx: dict[int, int] = {}
    for p, (i, j, _) in enumerate(pairs):
        by_i[i].append(p)
        by_j[j].append(p)
        if i == j:
            self_idx[j] = p
    for i in nodes:
        for p in by_i[i]:
            rows.append(r); cols.append(m + p); vals.append(1.0)
        lb.append(1.0); ub.append(1.0); r += 1
    # linking x_ij <= y_j (i != j) and x_jj = y_j
    for p, (i, j, _) in enumerate(pairs):
        rows += [r, r]; cols += [m + p, col[j]]; vals += [1.0, -1.0]
        if i == j:
            lb.append(0.0); ub.append(0.0)
        else:
            lb.append(-np.inf); ub.append(0.0)
        r += 1
    # soft capacity
    for j in nodes:
        for p in by_j[j]:
            i = pairs[p][0]
            rows.append(r); cols.append(m + p); vals.append(demand[i])
        rows += [r, r]; cols += [col[j], m + P + col[j]]; vals += [-capacity[j], -1.0]
        lb.append(-np.inf); ub.append(0.0); r += 1
    A = coo_matrix((vals, (rows, cols)), shape=(r, nv)).tocsr()
    integrality = np.concatenate([np.ones(m + P), np.zeros(m)])
    upper = np.concatenate([np.ones(m + P), np.full(m, np.inf)])
    res = milp(c, constraints=LinearConstraint(A, lb, ub), integrality=integrality,
               bounds=Bounds(np.zeros(nv), upper),
               options={"time_limit": time_limit, "mip_rel_gap": mip_gap, "disp": False})
    info = {"status": int(res.status), "time": time.perf_counter() - t0, "n_vars": nv,
            "n_cons": r, "obj": float(res.fun) if res.x is not None else math.nan,
            "gap": float(getattr(res, "mip_gap", math.nan) or 0.0) if res.x is not None else math.nan,
            "bound": float(getattr(res, "mip_dual_bound", math.nan)) if res.x is not None else math.nan}
    if res.x is None:
        return None, None, info
    x = res.x
    heads = [j for j in nodes if x[col[j]] > 0.5]
    ch_of = {}
    for p, (i, j, _) in enumerate(pairs):
        if x[m + p] > 0.5 and i != j:
            ch_of[i] = j
    info["overflow"] = float(x[m + P:].sum())
    return heads, ch_of, info


def greedy_capacitated(nodes, open_cost, pairs, demand, capacity):
    """Cost-effectiveness greedy for the same problem (used as a fallback and
    as a reference heuristic in the solver study)."""
    t0 = time.perf_counter()
    cand: dict[int, list[tuple[float, int]]] = {j: [] for j in nodes}
    for i, j, cij in pairs:
        cand[j].append((cij, i))
    for j in cand:
        cand[j].sort()
    unassigned = set(nodes)
    heads, ch_of = [], {}
    obj = 0.0
    while unassigned:
        best = None
        for j in nodes:
            if j in heads:
                continue
            load, cost, take = 0.0, open_cost[j], []
            for cij, i in cand[j]:
                if i in unassigned and load + demand[i] <= capacity[j] + 1e-9:
                    load += demand[i]
                    cost += cij
                    take.append(i)
            if j not in take and j in unassigned:
                continue
            if not take:
                continue
            ratio = cost / len(take)
            if best is None or ratio < best[0]:
                best = (ratio, j, take, cost)
        if best is None:  # capacity exhausted: remaining nodes become singleton heads
            for i in list(unassigned):
                heads.append(i)
                obj += open_cost[i]
                unassigned.discard(i)
            break
        _, j, take, cost = best
        heads.append(j)
        obj += cost
        for i in take:
            unassigned.discard(i)
            if i != j:
                ch_of[i] = j
    return heads, ch_of, {"time": time.perf_counter() - t0, "obj": obj}
