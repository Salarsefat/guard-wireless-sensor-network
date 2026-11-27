import sys, json; sys.path.insert(0, "dev"); sys.path.insert(0, "src")
import pandas as pd
from quick import run
V = {"G": {"route": "meas", "batch": "sqrt", "nlr": 2, "edf": True, "hdr": 0, "c0": 0.0, "kappa": 1.0},
     "I": {"route": "meas", "batch": "sqrt", "nlr": 2, "edf": True, "hdr": 0, "c0": 0.0, "kappa": 1.0, "ebal": 0.5},
     "K": {"route": "hybrid", "batch": "sqrt", "nlr": 2, "edf": True, "hdr": 0, "c0": 0.0, "kappa": 1.0, "rdel": 0.01}}
methods = ["bcp", "cetx", "twinloop", "ganaco"] + [("v2", v) for v in V.values()]
res = run(methods, [{"lam": 1.0, "T": 900.0, "E0": 0.3}], [3001, 3002, 3003])
inv = {json.dumps(v, sort_keys=True): k for k, v in V.items()}
for r in res:
    r["variant"] = inv.get(json.dumps(json.loads(r["params"]), sort_keys=True), r["method"]) if r["method"] == "v2" else r["method"]
df = pd.DataFrame(res); df.to_json("dev/life1.jsonl", orient="records", lines=True)
pd.set_option("display.width", 250)
print(df.groupby("variant")[["fnd", "hnd", "alive_frac", "tdr_1", "gdr", "energy_per_reading_mJ", "delivered"]].mean().round(3).to_string())
