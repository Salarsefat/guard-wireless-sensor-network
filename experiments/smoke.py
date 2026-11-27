"""Short smoke run: every method once on a small network (about 1 minute)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wsncc.config import METHODS  # noqa: E402
from wsncc.runner import run_one  # noqa: E402

if __name__ == "__main__":
    scen = {"n": 40, "area": 320.0, "lam": 1.0, "T": 60.0, "warm": 10.0, "drain": 5.0}
    print(f"{'method':10s} {'TDR(1s)':>8s} {'GDR':>6s} {'delay(ms)':>10s}")
    for m in METHODS:
        r = run_one(m, {}, scen, seed=424242, rl_train_scen={"n": 40, "area": 320.0, "T": 60.0})
        print(f"{m:10s} {r['tdr_1']:8.3f} {r['gdr']:6.3f} {1000 * r['delay_mean']:10.1f}")
