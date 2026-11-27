"""Shared helpers for loading results, statistics and plotting."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RES = ROOT / "results" / "test2"
OUT = ROOT / "output"            # generated figures and LaTeX tables
FIG = OUT / "figures"
TAB = OUT / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

METHODS = ["guard", "twinloop", "cetx", "bcp", "cbpr", "ganaco", "acsro", "fsfg", "rlcc"]
BASELINES = METHODS[1:]
LABEL = {"guard": "GUARD", "twinloop": "TwinLoop", "cetx": "C-ETX", "bcp": "BCP",
         "cbpr": "CBPR-like", "ganaco": "GAN-ACO", "acsro": "ACSRO-like", "fsfg": "FSFG-like",
         "rlcc": "RL-CC"}
# Methods drawn in figures (ACSRO-like and RL-CC coincide with C-ETX; they are in every table).
PLOT = ["guard", "twinloop", "cetx", "bcp", "cbpr", "ganaco", "fsfg"]
# validated categorical palette (dataviz validator: all checks pass in this order, light mode)
COLOR = dict(zip(PLOT, ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#008a17", "#6a4ce2"]))
COLOR.update({"acsro": "#8c8c8c", "rlcc": "#8c8c8c"})
MARKER = dict(zip(METHODS, ["o", "s", "^", "v", "D", "P", "X", "h", "*"]))
TEXT = "#191919"
TEXT2 = "#666666"
GRID = "#e6e6e6"


def load(exp: str, ok_only: bool = True) -> pd.DataFrame:
    path = RES / f"{exp}.jsonl"
    recs = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    df = pd.DataFrame(recs)
    if ok_only and "status" in df:
        bad = df[df["status"] != "ok"]
        if len(bad):
            print(f"[warn] {exp}: {len(bad)} failed runs", file=sys.stderr)
        df = df[df["status"] == "ok"].copy()
    return df


def mean_ci(x, conf: float = 0.95):
    x = np.asarray([v for v in x if v == v], dtype=float)
    n = len(x)
    if n == 0:
        return math.nan, math.nan, 0
    m = float(x.mean())
    if n < 2:
        return m, math.nan, n
    h = float(stats.t.ppf(0.5 + conf / 2, n - 1) * x.std(ddof=1) / math.sqrt(n))
    return m, h, n


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for k, idx in enumerate(order):
        val = min(1.0, (m - k) * p[idx])
        running = max(running, val)
        adj[idx] = running
    return adj


def boot_ci_median(d, n_boot: int = 10_000, seed: int = 0):
    d = np.asarray(d, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (n_boot, len(d)))
    meds = np.median(d[idx], axis=1)
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def paired_tests(df: pd.DataFrame, metric: str, group: str, ref: str = "guard",
                 methods=None) -> pd.DataFrame:
    """Wilcoxon signed-rank (two-sided) ref vs each baseline within each group
    value, paired by seed; Holm correction across baselines within the group."""
    methods = [m for m in (methods or METHODS) if m != ref]
    rows = []
    for gval, sub in df.groupby(group):
        piv = sub.pivot_table(index="seed", columns="method", values=metric, aggfunc="first")
        fam = []
        for m in methods:
            if ref not in piv or m not in piv:
                continue
            pair = piv[[ref, m]].dropna()
            d = (pair[ref] - pair[m]).to_numpy()
            n = len(d)
            if n == 0:
                continue
            nz = d[d != 0]
            if len(nz) == 0:
                p, wplus, wminus = 1.0, 0.0, 0.0
            else:
                res = stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided")
                p = float(res.pvalue)
                ranks = stats.rankdata(np.abs(nz))
                wplus = float(ranks[nz > 0].sum())
                wminus = float(ranks[nz < 0].sum())
            rbc = (wplus - wminus) / (wplus + wminus) if wplus + wminus > 0 else 0.0
            lo, hi = boot_ci_median(d) if n > 1 else (math.nan, math.nan)
            fam.append({group: gval, "metric": metric, "baseline": m, "n_pairs": n,
                        "mean_ref": float(pair[ref].mean()), "mean_base": float(pair[m].mean()),
                        "median_diff": float(np.median(d)), "ci_lo": lo, "ci_hi": hi,
                        "rank_biserial": rbc, "p": p})
        if fam:
            adj = holm([r["p"] for r in fam])
            for r, a in zip(fam, adj):
                r["p_holm"] = float(a)
            rows += fam
    return pd.DataFrame(rows)


def friedman(df: pd.DataFrame, metric: str, group: str, higher_better: bool = True) -> pd.DataFrame:
    rows = []
    for gval, sub in df.groupby(group):
        piv = sub.pivot_table(index="seed", columns="method", values=metric, aggfunc="first")
        piv = piv[[m for m in METHODS if m in piv]].dropna()
        if len(piv) < 3:
            continue
        stat, p = stats.friedmanchisquare(*[piv[c].to_numpy() for c in piv.columns])
        r = piv.rank(axis=1, ascending=not higher_better).mean()
        row = {group: gval, "metric": metric, "chi2": float(stat), "p": float(p), "n": len(piv)}
        row.update({f"rank_{m}": float(r[m]) for m in piv.columns})
        rows.append(row)
    return pd.DataFrame(rows)


def setup_mpl():
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5,
        "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "legend.fontsize": 7.2, "axes.edgecolor": TEXT2, "axes.labelcolor": TEXT,
        "xtick.color": TEXT2, "ytick.color": TEXT2, "text.color": TEXT,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.4,
        "lines.markersize": 4.5, "savefig.bbox": "tight", "savefig.dpi": 300,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
