import sys, json; sys.path.insert(0, "dev"); sys.path.insert(0, "src")
import pandas as pd
from quick import run, KEYS
methods = ["cetx", "cbpr", "ganaco", "bcp", "twinloop"]
scens = [{"lam": l, "T": 150.0} for l in (0.25, 1.0, 2.0, 3.0)]
res = run(methods, scens, [3001, 3002, 3003, 3004])
df = pd.DataFrame(res)
df.to_json("dev/diag1.jsonl", orient="records", lines=True)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
cols = ["tdr_1", "gdr", "delay_mean", "delay_p95", "energy_per_reading_mJ", "jain", "readings_per_frame", "hops_mean", "coll_frac_all", "sink_coll_frac", "drop_retry", "drop_caf", "drop_buf"]
print(df.groupby(["lam", "method"])[cols].mean().round(3).to_string())
