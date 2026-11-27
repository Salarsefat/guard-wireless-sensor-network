"""Generic DEV round: python dev/roundn.py <name> <json variants> [loads] [seeds] [T]"""
import sys, json; sys.path.insert(0, "dev"); sys.path.insert(0, "src")
import pandas as pd
from quick import run
name = sys.argv[1]
V = json.loads(open(sys.argv[2]).read()) if sys.argv[2].endswith(".json") else json.loads(sys.argv[2])
loads = [float(x) for x in (sys.argv[3].split(",") if len(sys.argv) > 3 else ["0.25", "1", "2", "3"])]
seeds = [int(x) for x in (sys.argv[4].split(",") if len(sys.argv) > 4 else ["3001", "3002", "3003", "3004"])]
T = float(sys.argv[5]) if len(sys.argv) > 5 else 150.0
extra = json.loads(sys.argv[6]) if len(sys.argv) > 6 else {}
methods = [("v2", v) for v in V.values()]
scens = [{"lam": l, "T": T, **extra} for l in loads]
res = run(methods, scens, seeds)
inv = {json.dumps(v, sort_keys=True): k for k, v in V.items()}
for r in res:
    r["variant"] = inv[json.dumps(json.loads(r["params"]), sort_keys=True)]
df = pd.DataFrame(res); df.to_json(f"dev/{name}.jsonl", orient="records", lines=True)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
cols = ["tdr_1", "gdr", "delay_mean", "delay_p95", "delay_p99", "energy_per_reading_mJ", "jain", "readings_per_frame", "hops_mean", "coll_frac_all", "sink_coll_frac", "drop_retry", "drop_caf", "fnd"]
print(df.groupby(["lam", "variant"])[cols].mean().round(3).to_string())
