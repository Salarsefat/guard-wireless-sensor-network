"""Equal-budget random search on the VALIDATION seeds only.

Every method receives TUNING_BUDGET configurations: its default (published)
setting plus TUNING_BUDGET - 1 configurations drawn uniformly at random from its
search space (config.SEARCH_SPACES) with a fixed RNG seed.  Each configuration
is evaluated on the same validation seeds and offered loads; the objective is
the mean timely delivery ratio TDR(1 s).  Configurations within TIE_TOL of the
best mean TDR are tied; ties are broken by the lowest mean 95th-percentile
delay, then by the mean delivery ratio (docs/PROTOCOL_v2.md, section 3).
RL-CC Q-tables are trained on the TRAIN seeds only.

Usage:  python experiments/tune.py [--workers 2]
Outputs results/tuning/tuning_runs.jsonl and results/tuned_params.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from batch import run_batch  # noqa: E402
from wsncc.config import (METHODS, SEARCH_SPACES, TIE_TOL, TUNE_LOADS, TUNE_T,  # noqa: E402
                          TUNING_BUDGET, VAL_SEEDS)

TUNE_SEEDS = VAL_SEEDS[:2]


def configurations(method: str, budget: int = TUNING_BUDGET, seed: int = 12345) -> list[dict]:
    rng = random.Random(f"{seed}-{method}")
    space = SEARCH_SPACES[method]
    confs = [{}]                                       # default / published setting first
    seen = {json.dumps({}, sort_keys=True)}
    tries = 0
    while len(confs) < budget and tries < 10_000:
        tries += 1
        c = {k: rng.choice(v) for k, v in space.items()}
        key = json.dumps(c, sort_keys=True)
        if key not in seen:
            seen.add(key)
            confs.append(c)
    return confs


def jobs_for(method: str) -> list[dict]:
    jobs = []
    for ci, conf in enumerate(configurations(method)):
        for lam in TUNE_LOADS:
            for seed in TUNE_SEEDS:
                jobs.append({"method": method, "params": conf, "seed": seed,
                             "scen": {"lam": lam, "T": TUNE_T}, "tag": f"tune:{method}:{ci}",
                             "rl_train_scen": {"T": TUNE_T}})
    return jobs


def select(path: Path) -> dict:
    recs = [json.loads(x) for x in path.read_text().splitlines()]
    recs = [r for r in recs if r.get("status") == "ok" and r["tag"].startswith("tune:")]
    best = {}
    table = {}
    for r in recs:
        _, m, ci = r["tag"].split(":")
        table.setdefault(m, {}).setdefault(int(ci), []).append(r)
    for m, confs in table.items():
        scored = []
        n_expected = len(TUNE_LOADS) * len(TUNE_SEEDS)
        for ci, rs in confs.items():
            if len(rs) < n_expected:
                continue
            tdr = sum(x["tdr_1"] for x in rs) / len(rs)
            gdr = sum(x["gdr"] for x in rs) / len(rs)
            p95 = sum(x["delay_p95"] for x in rs) / len(rs)
            scored.append({"ci": ci, "tdr_1": tdr, "gdr": gdr, "delay_p95": p95,
                           "params": json.loads(rs[0]["params"]), "runs": len(rs)})
        top = max(x["tdr_1"] for x in scored)
        tied = [x for x in scored if x["tdr_1"] >= top - TIE_TOL]
        tied.sort(key=lambda x: (x["delay_p95"], -x["gdr"]))
        win = tied[0]
        scored.sort(key=lambda x: x["tdr_1"], reverse=True)
        best[m] = {"params": win["params"], "config_index": win["ci"], "tdr_1": win["tdr_1"],
                   "gdr": win["gdr"], "delay_p95": win["delay_p95"], "runs": win["runs"],
                   "n_tied": len(tied), "best_tdr_1": top, "ranking": scored}
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--budget", type=float, default=None)
    a = ap.parse_args()
    out = ROOT / "results" / "tuning" / "tuning_runs.jsonl"
    jobs = []
    for m in a.methods.split(","):
        jobs += jobs_for(m)
    run_batch(jobs, out, a.workers, a.budget)
    best = select(out)
    (ROOT / "results" / "tuned_params.json").write_text(json.dumps(best, indent=2))
    for m, b in best.items():
        print(f"{m:9s} TDR1={b['tdr_1']:.4f} GDR={b['gdr']:.4f} config#{b['config_index']} {b['params']}")


if __name__ == "__main__":
    main()
