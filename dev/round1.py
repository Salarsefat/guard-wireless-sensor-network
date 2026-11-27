import sys, json; sys.path.insert(0, "dev"); sys.path.insert(0, "src")
import pandas as pd
from quick import run
V = {
 "flat":      {"route": "etx", "batch": "none", "nlr": 0, "edf": False},
 "flat+nlr":  {"route": "etx", "batch": "none", "nlr": 2, "edf": False},
 "ht+nlr":    {"route": "ht",  "batch": "none", "nlr": 2, "edf": False},
 "ht+nlr+b":  {"route": "ht",  "batch": "sqrt", "nlr": 2, "edf": False},
 "full":      {"route": "ht",  "batch": "sqrt", "nlr": 2, "edf": True},
}
methods = [("v2", v) for v in V.values()]
scens = [{"lam": l, "T": 150.0} for l in (0.25, 1.0, 2.0, 3.0)]
res = run(methods, scens, [3001, 3002, 3003, 3004])
inv = {json.dumps(v, sort_keys=True): k for k, v in V.items()}
for r in res:
    r["variant"] = inv[json.dumps(json.loads(r["params"]), sort_keys=True)]
df = pd.DataFrame(res); df.to_json("dev/round1.jsonl", orient="records", lines=True)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
cols = ["tdr_1", "gdr", "delay_mean", "delay_p95", "energy_per_reading_mJ", "jain", "readings_per_frame", "hops_mean", "coll_frac_all", "sink_coll_frac", "drop_retry", "drop_caf"]
print(df.groupby(["lam", "variant"])[cols].mean().round(3).to_string())
