"""Regression: baselines must be unchanged by the behaviour-neutral hooks."""
import sys, json; sys.path.insert(0, "dev"); sys.path.insert(0, "src")
import pandas as pd
from quick import run
old = pd.read_json("dev/diag1.jsonl", lines=True)
methods = ["cetx", "cbpr", "ganaco", "bcp", "twinloop"]
res = pd.DataFrame(run(methods, [{"lam": 2.0, "T": 150.0}], [3001, 3002]))
o = old[(old.lam == 2.0) & old.seed.isin([3001, 3002])]
m = res.merge(o, on=["method", "seed"], suffixes=("_new", "_old"))
for k in ["tdr_1", "gdr", "delay_mean", "energy_per_reading_mJ", "tx_frames"]:
    d = (m[k + "_new"] - m[k + "_old"]).abs().max()
    print(k, "max abs diff", d)
