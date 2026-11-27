import sys, json; sys.path.insert(0, "dev"); sys.path.insert(0, "src")
import pandas as pd
from quick import run
base = {"route": "meas", "batch": "sqrt", "nlr": 2, "edf": True, "hdr": 0, "c0": 0.0, "kappa": 1.0}
V = {"I_eb05": {**base, "ebal": 0.5},
     "L1_eb05_cE05": {**base, "ebal": 0.5, "cE": 0.5},
     "L2_eb05_cE2": {**base, "ebal": 0.5, "cE": 2.0},
     "L3_eb1_cE1": {**base, "ebal": 1.0, "cE": 1.0}}
methods = [("v2", v) for v in V.values()]
res = run(methods, [{"lam": 1.0, "T": 900.0, "E0": 0.3}], [3001, 3002, 3003])
inv = {json.dumps(v, sort_keys=True): k for k, v in V.items()}
for r in res:
    r["variant"] = inv[json.dumps(json.loads(r["params"]), sort_keys=True)]
df = pd.DataFrame(res); df.to_json("dev/life2.jsonl", orient="records", lines=True)
print(df.groupby("variant")[["fnd", "alive_frac", "tdr_1", "gdr", "energy_per_reading_mJ", "delivered", "readings_per_frame", "delay_mean", "delay_p95"]].mean().round(3).to_string())
