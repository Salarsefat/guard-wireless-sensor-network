"""Final experiments E1-E8 of version 2 on the TEST2 seeds with frozen (tuned) settings.

Usage:
    python experiments/run_experiments.py E1 [E2 ...] [--workers 2]
Reads results/tuned_params.json (produced by tune.py) and writes one JSON-lines
file per experiment to results/test2/.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from batch import run_batch  # noqa: E402
from wsncc.config import METHODS, TEST_SEEDS  # noqa: E402

E1_LOADS = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]


def tuned() -> dict:
    t = json.loads((ROOT / "results" / "tuned_params.json").read_text())
    return {m: t[m]["params"] for m in METHODS}


def _jobs(methods, params, scen_list, seeds, tag, timeseries_seed=None):
    jobs = []
    for sc in scen_list:
        for m in methods:
            for s in seeds:
                jobs.append({"method": m, "params": params[m], "scen": sc, "seed": s, "tag": tag,
                             "rl_train_scen": {"T": 200.0},
                             "timeseries": timeseries_seed is not None and s == timeseries_seed})
    return jobs


def build(exp: str) -> list[dict]:
    """Experiments of docs/PROTOCOL_v2.md (TEST2 seeds 101-130, run once)."""
    P = tuned()
    S10 = TEST_SEEDS[:10]
    if exp == "E1":
        scen = [{"lam": lam, "T": 300.0} for lam in E1_LOADS]
        return _jobs(METHODS, P, scen, TEST_SEEDS[:15], "E1", timeseries_seed=TEST_SEEDS[0])
    if exp == "E2":
        scen = [{"n": n, "area": round(500.0 * math.sqrt(n / 100.0), 1), "lam": 1.0, "T": 300.0}
                for n in (50, 100, 200, 300)]
        return _jobs(METHODS, P, scen, S10, "E2")
    if exp == "E3":
        variants = {"corner": {"sink": "corner"}, "clustered": {"topo": "clustered"},
                    "events": {"events": True, "lam": 0.5}, "cc2420": {"energy": "cc2420"},
                    "harsh": {"capture_delta": 1.0}, "mild": {"capture_delta": 0.3}}
        jobs = []
        for name, v in variants.items():
            sc = {"lam": 1.0, "T": 300.0, **v}
            jobs += _jobs(METHODS, P, [sc], S10, f"E3:{name}")
        return jobs
    if exp == "E4":
        scen = [{"lam": 1.0, "T": 900.0, "E0": 0.3}]
        return [{**j, "timeseries": True} for j in _jobs(METHODS, P, scen, S10, "E4")]
    if exp == "E5":
        base = P["guard"]
        variants = {
            "full": {},
            "no_dispatch": {"batch": False},
            "no_nlr": {"nlr": 0},
            # Amendment A1: the tuned GUARD has rdel = 0, so "no_decorrelation" would equal "full"
            "with_decorrelation": {"rdel": 0.01},
            "no_edf": {"edf": False},
            "static_etx": {"route": "etx"},
            "no_energy_terms": {"ebal": 0.0, "cE": 0.0, "cR": 0.0},
            "model_prior": {"route": "hybrid"},
            "ctp_like": {"route": "meas", "batch": False, "edf": False, "rdel": 0.0,
                         "ebal": 0.0, "cE": 0.0, "cR": 0.0},
        }
        jobs = []
        for name, v in variants.items():
            params = {"guard": {**base, **v}}
            jobs += _jobs(["guard"], params, [{"lam": lam, "T": 300.0} for lam in (0.5, 1.0, 2.0, 3.0)],
                          S10, f"E5:{name}")
        return jobs
    if exp == "E6":
        base = P["guard"]
        grid = {"kappa": [0.25, 0.5, 1.0, 2.0, 4.0], "beta": [0.05, 0.1, 0.25, 0.5, 0.75],
                "alpha_s": [0.02, 0.05, 0.1, 0.2], "nlr": [0, 1, 2, 3, 4],
                # Amendment A1: tuned rdel and u0 lie at the edge of their search ranges
                "rdel": [0.0, 0.005, 0.01, 0.02, 0.05], "u0": [0.0, 0.1, 0.2, 0.3, 0.4]}
        jobs = []
        for k, vals in grid.items():
            for v in vals:
                params = {"guard": {**base, k: v}}
                jobs += _jobs(["guard"], params, [{"lam": 2.0, "T": 300.0}], S10, f"E6:{k}={v}")
        return jobs
    if exp == "E7":
        # energy-latency frontier of GUARD: holding-cost weight h (batch=False = no holding)
        base = P["guard"]
        jobs = []
        for h in (0.0025, 0.005, 0.01, 0.02, 0.04, 0.08):
            params = {"guard": {**base, "h": h}}
            jobs += _jobs(["guard"], params, [{"lam": lam, "T": 300.0} for lam in (1.0, 2.0)],
                          S10, f"E7:h={h}")
        params = {"guard": {**base, "batch": False}}
        jobs += _jobs(["guard"], params, [{"lam": lam, "T": 300.0} for lam in (1.0, 2.0)],
                      S10, "E7:h=inf")
        return jobs
    if exp == "E8":
        # default (untuned) configuration of every method
        default = {m: {} for m in METHODS}
        scen = [{"lam": lam, "T": 300.0} for lam in (0.5, 1.0, 2.0)]
        return _jobs(METHODS, default, scen, S10, "E8")
    raise ValueError(exp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiments", nargs="+")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--budget", type=float, default=None)
    a = ap.parse_args()
    for exp in a.experiments:
        jobs = build(exp)
        out = ROOT / "results" / "test2" / f"{exp}.jsonl"
        print(f"== {exp}: {len(jobs)} jobs -> {out}", flush=True)
        run_batch(jobs, out, a.workers, a.budget)


if __name__ == "__main__":
    main()
