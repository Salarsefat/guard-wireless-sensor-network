import sys, json; sys.path.insert(0, "dev"); sys.path.insert(0, "src")
import pandas as pd
from quick import run
res = run(["guard"], [{"lam": l, "T": 150.0} for l in (0.25, 1.0, 2.0, 3.0)], [3001, 3002, 3003, 3004])
df = pd.DataFrame(res); df.to_json("dev/guard_check.jsonl", orient="records", lines=True)
cols = ["tdr_1", "gdr", "delay_mean", "delay_p95", "delay_p99", "energy_per_reading_mJ", "jain", "readings_per_frame", "hops_mean", "coll_frac_all", "sink_coll_frac", "drop_retry", "frames_per_reading", "dec_us", "wall"]
pd.set_option("display.width", 250)
print(df.groupby("lam")[cols].mean().round(4).to_string())
