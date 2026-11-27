"""Objective-by-objective scorecard of GUARD against every baseline (all TEST2 experiments).

Reads the paired-test files written by v2.py (results/test2/paired_tests_v2_E*.csv) and
counts, for every metric, the comparisons (experiment condition x baseline) in which
GUARD was significantly better, not significantly different, or significantly worse
(two-sided Wilcoxon signed-rank, Holm over the eight baselines, alpha = 0.05).
Output: results/test2/scorecard.md and results/test2/scorecard.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "test2"

# metric -> (objective as named in the study's brief, higher is better)
OBJ = {
    "tdr_1": ("timely delivery ratio TDR(1 s) (in E4 the denominators differ between methods)", True),
    "tdr_05": ("timely delivery ratio TDR(0.5 s)", True),
    "tdr_2": ("timely delivery ratio TDR(2 s)", True),
    "gdr": ("packet delivery ratio", True),
    "throughput_rps": ("throughput", True),
    "jain": ("fairness (Jain index)", True),
    "lost_frac": ("packet loss ratio", False),
    "delay_mean": ("mean delay", False),
    "delay_p95": ("95th-percentile delay", False),
    "delay_p99": ("99th-percentile delay", False),
    "energy_per_reading_mJ": ("energy per delivered reading", False),
    "coll_frac_all": ("collided frames (all receivers)", False),
    "sink_coll_frac": ("collided frames (sink)", False),
    "frames_per_reading": ("frames per delivered reading", False),
    "ctrl_per_s": ("control bytes", False),
    "ovh_per_s": ("total protocol overhead bytes", False),
    "fnd_f": ("lifetime: first node dead", True),
    "alive_frac": ("lifetime: alive fraction at 900 s", True),
    "timely": ("lifetime: timely readings delivered", True),
    "delivered": ("lifetime: readings delivered", True),
    "collapse90": ("lifetime: time with >= 90 % delivery", True),
    "collapse50": ("lifetime: time with >= 50 % delivery", True),
    "gdr_crit": ("critical-reading delivery ratio (events)", True),
    "delay_crit_p95": ("critical-reading 95th-percentile delay (events)", False),
}


def main() -> None:
    frames = []
    for f in sorted(RES.glob("paired_tests_v2_E*.csv")):
        df = pd.read_csv(f)
        df["exp"] = f.stem.split("_")[-1]
        frames.append(df)
    allt = pd.concat(frames, ignore_index=True)
    rows = []
    for metric, (name, hb) in OBJ.items():
        t = allt[allt["metric"] == metric]
        if not len(t):
            continue
        sig = t["p_holm"] < 0.05
        better = (t["median_diff"] > 0) == hb
        win, loss = int((sig & better).sum()), int((sig & ~better).sum())
        tie = int(len(t) - win - loss)
        losers = sorted(set(t.loc[sig & ~better, "baseline"]))
        rows.append({"objective": name, "metric": metric, "experiments": ",".join(sorted(set(t["exp"]))),
                     "comparisons": int(len(t)), "win": win, "tie": tie, "loss": loss,
                     "baselines_better": ", ".join(losers)})
    out = pd.DataFrame(rows)
    (RES / "scorecard.json").write_text(json.dumps(rows, indent=1))
    lines = ["| Objective | Experiments | Comparisons | GUARD better | n.s. | GUARD worse | Baselines significantly better |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['objective']} | {r['experiments']} | {r['comparisons']} | {r['win']} | {r['tie']} | "
                     f"{r['loss']} | {r['baselines_better'] or '-'} |")
    (RES / "scorecard.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
