"""Cost of GUARD's per-second route update (instrumentation only).

The simulator times cluster formation (Protocol.recluster) as central computation,
but GUARD updates its routes inside control(), which is not timed.  This script
wraps GUARD._route with a timer (no behaviour change) and reports the mean
wall-clock time per update, i.e., per simulated second, for N = 100, 200, 300
sensors at constant density, TEST2 seeds 101-103, 1 reading/s per node, T = 60 s.
Output: results/test2/route_timing.json
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    from wsncc.protocols import make_protocol
    from wsncc.runner import scenario
    from wsncc.sim import Sim
    tuned = json.loads((ROOT / "results" / "tuned_params.json").read_text())["guard"]["params"]
    out = {}
    for n in (100, 200, 300):
        area = round(500.0 * math.sqrt(n / 100.0), 1)
        per = []
        for seed in (101, 102, 103):
            proto = make_protocol("guard", dict(tuned))
            sim = Sim(scenario(n=n, area=area, lam=1.0, T=60.0), proto, seed)
            orig = proto._route
            acc = [0.0, 0]

            def timed(orig=orig, acc=acc):
                t0 = time.perf_counter()
                orig()
                acc[0] += time.perf_counter() - t0
                acc[1] += 1

            proto._route = timed
            sim.run()
            per.append(1000.0 * acc[0] / max(acc[1], 1))
        out[str(n)] = {"ms_per_update": float(np.mean(per)), "per_seed": per}
        print(n, round(float(np.mean(per)), 2), "ms per route update")
    (ROOT / "results" / "test2" / "route_timing.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
