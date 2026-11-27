import sys, json; sys.path.insert(0, "dev"); sys.path.insert(0, "src")
import pandas as pd
from quick import run
base = {"route": "meas", "batch": "sqrt", "nlr": 2, "edf": True, "hdr": 0, "c0": 0.0, "kappa": 1.0, "ebal": 0.5, "cE": 2.0}
V = {"L2": base, "R05": {**base, "cR": 0.5}, "R2": {**base, "cR": 2.0}}
methods = [("v2", v) for v in V.values()]
scen = sys.argv[1]
sc = {"life": {"lam": 1.0, "T": 900.0, "E0": 0.3}, "main1": {"lam": 1.0, "T": 300.0}, "main2": {"lam": 2.0, "T": 300.0}}[scen]
res = run(methods, [sc], [3001, 3002, 3003])
inv = {json.dumps(v, sort_keys=True): k for k, v in V.items()}
for r in res:
    r["variant"] = inv[json.dumps(json.loads(r["params"]), sort_keys=True)]
df = pd.DataFrame(res); df.to_json(f"dev/life3_{scen}.jsonl", orient="records", lines=True)
print(scen); print(df.groupby("variant")[["fnd", "alive_frac", "tdr_1", "gdr", "energy_per_reading_mJ", "delivered", "readings_per_frame", "delay_mean", "delay_p95"]].mean().round(3).to_string())
