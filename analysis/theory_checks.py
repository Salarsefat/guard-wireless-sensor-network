"""Numerical checks of the analytical results (no network simulation).

1. Theorem 1: average cost of the control-limit dispatch policy under Poisson
   arrivals, g(m) = c*Lam/m + h*(m-1)/2, and the closed-form minimiser
   m* = min{m : m(m+1) >= 2 c Lam / h}, against Monte Carlo.
2. Lemma 1 / Proposition 1: hidden part of the guard zone, analytic vs Monte Carlo.
3. Proposition 3: re-collision probability of two hidden frames after
   independent U(0, b) delays, analytic bound 2A/b - (A/b)^2 vs Monte Carlo, and
   the IEEE 802.15.4 retry backoff (BE = 3) for comparison.
Writes results/theory/theory_checks.json.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "theory"


def g(m, c, lam, h):
    return c * lam / m + h * (m - 1) / 2.0


def m_star(c, lam, h, K):
    x = 2.0 * c * lam / h
    m = 1
    while m < K and m * (m + 1) < x:
        m += 1
    return m


def mc_cost(m, c, lam, h, n_cycles=200_000, rng=None):
    """Monte Carlo of the control-limit policy: renewal cycles of m arrivals."""
    rng = rng or np.random.default_rng(1)
    gaps = rng.exponential(1.0 / lam, size=(n_cycles, m))
    # reading j (1-based) waits sum of gaps j+1..m ; holding cost h * total waiting
    wait = np.zeros(n_cycles)
    for j in range(1, m):
        wait += j * gaps[:, j]
    cycle = gaps.sum(axis=1)
    return float((c * n_cycles + h * wait.sum()) / cycle.sum())


def lens(a, b, d):
    if d >= a + b:
        return 0.0
    if d <= abs(a - b):
        return math.pi * min(a, b) ** 2
    t1 = a * a * math.acos((d * d + a * a - b * b) / (2 * d * a))
    t2 = b * b * math.acos((d * d + b * b - a * a) / (2 * d * b))
    t3 = 0.5 * math.sqrt((-d + a + b) * (d + a - b) * (d - a + b) * (d + a + b))
    return t1 + t2 - t3


def hidden_area(d, delta, rcs):
    a = (1 + delta) * d
    return math.pi * a * a - lens(a, rcs, d)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    res = {"theorem1": [], "lemma1": [], "prop3": {}}
    K = 4
    h = 0.01
    A = 0.0032                      # airtime of a full frame [s]
    for lam in (2.0, 10.0, 25.0, 80.0):
        for price in (0.25, 1.0, 4.0):
            c = A * price
            row = {"lam": lam, "price": price, "m_star": m_star(c, lam, h, K)}
            costs = {}
            for m in range(1, K + 1):
                costs[m] = {"analytic": g(m, c, lam, h), "mc": mc_cost(m, c, lam, h, rng=rng)}
            row["costs"] = costs
            row["m_mc_argmin"] = int(min(costs, key=lambda k: costs[k]["mc"]))
            row["max_rel_err"] = max(abs(v["mc"] - v["analytic"]) / v["analytic"] for v in costs.values())
            res["theorem1"].append(row)
    delta, rcs = 0.6, 150.0
    for d in (20, 40, 57.69, 60, 70, 80, 90, 100, 110, 120):
        a = (1 + delta) * d
        pts = rng.uniform(-a, a, size=(400_000, 2))
        inz = (pts ** 2).sum(1) < a * a
        hid = inz & (((pts - np.array([d, 0.0])) ** 2).sum(1) > rcs ** 2)
        res["lemma1"].append({"d": d, "analytic": hidden_area(d, delta, rcs),
                              "mc": float(hid.mean() * (2 * a) ** 2),
                              "zone": math.pi * a * a})
    res["lemma1_threshold"] = rcs / (2 + delta)
    # Proposition 3: re-collision of two hidden frames of airtime A
    A = 0.0033
    n = 400_000
    out = {}
    for b in (0.0, 0.01, 0.02, 0.03, 0.05):
        delta0 = rng.uniform(-A, A, n)           # offset of the colliding pair
        # IEEE 802.15.4 retry: ACK wait 0.864 ms, backoff U{0..7} x 0.32 ms, then CCA+turnaround
        mac1 = rng.integers(0, 8, n) * 0.00032
        mac2 = rng.integers(0, 8, n) * 0.00032
        x1 = rng.uniform(0, b, n) if b > 0 else 0.0
        x2 = rng.uniform(0, b, n) if b > 0 else 0.0
        off = delta0 + (mac1 + x1) - (mac2 + x2)
        out[str(b)] = {"mc": float(np.mean(np.abs(off) < A)),
                       "bound": (2 * A / b - (A / b) ** 2) if b >= 2 * A else None}
    res["prop3"] = out
    (OUT / "theory_checks.json").write_text(json.dumps(res, indent=1))
    for r in res["theorem1"]:
        print(f"Lam={r['lam']:5.1f} price={r['price']:4.2f}  m*={r['m_star']}  MC argmin={r['m_mc_argmin']}  max rel err={r['max_rel_err']:.4f}")
    for r in res["lemma1"]:
        print(f"d={r['d']:6.2f} hidden area analytic={r['analytic']:9.1f}  MC={r['mc']:9.1f}")
    for b, v in out.items():
        print(f"b={b}: P(re-collision) MC={v['mc']:.3f} bound={v['bound']}")


if __name__ == "__main__":
    main()
