"""Write the configuration of the study to experiments/configs/*.json.

The JSON files are a readable copy of the settings that the code uses; they are
generated from the single source of truth (src/wsncc/config.py, the protocol
classes and experiments/run_experiments.py) and are not read by any script.

Usage:  python experiments/export_configs.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from wsncc import config as C  # noqa: E402
from wsncc.protocols import make_protocol  # noqa: E402

OUT = ROOT / "experiments" / "configs"


def dump(name: str, obj) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=1) + "\n")
    print("written", OUT / name)


def main() -> None:
    dump("scenario_defaults.json", C.SCENARIO_DEFAULTS)
    dump("seed_sets.json", {
        "train": C.TRAIN_SEEDS, "dev": C.DEV_SEEDS, "val": C.VAL_SEEDS, "test2": C.TEST2_SEEDS,
        "retired_test_v1": C.RETIRED_TEST_SEEDS,
        "used": {"train (RL-CC Q-table)": C.TRAIN_SEEDS[:3],
                 "dev (design of GUARD, diagnosis of the baselines)": C.DEV_SEEDS[:4],
                 "val (tuning of every method)": C.VAL_SEEDS[:2],
                 "test2 E1": C.TEST2_SEEDS[:15], "test2 E2-E8": C.TEST2_SEEDS[:10]}})
    dump("search_spaces.json", {"budget": C.TUNING_BUDGET, "loads": C.TUNE_LOADS, "T": C.TUNE_T,
                                "tie_tolerance_tdr": C.TIE_TOL, "spaces": C.SEARCH_SPACES})
    dump("method_defaults.json", {m: make_protocol(m, {}).p for m in C.METHODS})
    from run_experiments import build   # needs results/tuned_params.json
    exps = {}
    for exp in ("E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8"):
        jobs = build(exp)
        scen = sorted({json.dumps(j["scen"], sort_keys=True) for j in jobs})
        exps[exp] = {"n_runs": len(jobs), "methods": sorted({j["method"] for j in jobs}),
                     "seeds": sorted({j["seed"] for j in jobs}),
                     "tags": sorted({j["tag"] for j in jobs}),
                     "scenarios": [json.loads(s) for s in scen]}
    dump("experiments.json", exps)


if __name__ == "__main__":
    main()
