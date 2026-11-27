"""Which sensors die first in E4?  (Instrumentation only; no design decision depends on it.)

Re-runs the E4 configuration (E0 = 0.3 J, 1 reading/s per node, T = 900 s) for
GUARD, BCP, TwinLoop v1 and C-ETX with their tuned configurations on TEST2
seeds 101-103 and records, for every node death, the time, the node's distance
to the sink and whether the node was a one-hop neighbour of the sink
(distance <= radio range R).  The simulator is wrapped, not modified.
Output: results/test2/lifetime_deaths.jsonl (one record per run; re-runs skip finished runs).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

METHODS = ["guard", "bcp", "twinloop", "cetx"]
SEEDS = [101, 102, 103]
# GUARD with its energy terms switched off (diagnostic variants, as in the E5 ablation)
VARIANTS = {"guard_no_route_energy": {"ebal": 0.0}, "guard_no_price_energy": {"cE": 0.0},
            "guard_no_energy_terms": {"ebal": 0.0, "cE": 0.0}}


def run(method: str, seed: int) -> dict:
    from wsncc.protocols import make_protocol
    from wsncc.runner import scenario
    from wsncc.sim import Sim
    tuned = json.loads((ROOT / "results" / "tuned_params.json").read_text())
    base = "guard" if method in VARIANTS else method
    params = dict(tuned[base]["params"])
    params.update(VARIANTS.get(method, {}))
    proto = make_protocol(base, params)
    sc = scenario(lam=1.0, T=900.0, E0=0.3)
    sim = Sim(sc, proto, seed)
    R = float(sc["R"])
    deaths = []
    orig = sim.kill

    def wrapped(i):
        if sim.alive[i]:
            d = float(np.hypot(*(np.asarray(sim.pos[i]) - np.asarray(sim.pos[sim.sink]))))
            deaths.append({"t": float(sim.now), "node": int(i), "d_sink": d})
        return orig(i)

    sim.kill = wrapped
    res = sim.run()
    ts = np.asarray(sim.ts_delivered if hasattr(sim, "ts_delivered") else [], dtype=float)
    d_all = np.array([np.hypot(*(np.asarray(sim.pos[i]) - np.asarray(sim.pos[sim.sink]))) for i in range(sim.n)])
    # alive sensors that still have a finite-cost path to the sink at the end of the run
    from wsncc.topology import etx_distances
    H, _ = etx_distances(sim.top, sim.alive) if hasattr(sim, "top") else (None, None)
    connected = None
    if H is not None:
        connected = int(sum(1 for i in range(sim.n) if sim.alive[i] and np.isfinite(H[i])))
    return {"method": method, "seed": seed, "R": R, "n_sink_nbrs": int((d_all <= R).sum()),
            "deaths": deaths, "fnd": res["fnd"], "alive_frac": res["alive_frac"],
            "alive_end": int(sum(sim.alive[:sim.n])), "alive_connected_end": connected,
            "delivered": res["delivered"], "timely": int(round(res["tdr_1"] * res["gen"]))}


def main() -> None:
    path = ROOT / "results" / "test2" / "lifetime_deaths.jsonl"
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["method"], r["seed"]))
    for m in METHODS + list(VARIANTS):
        for s in SEEDS:
            if (m, s) in done:
                continue
            r = run(m, s)
            with path.open("a") as fh:
                fh.write(json.dumps(r) + "\n")
            print(m, s, "done", flush=True)
    out = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    for r in out:
        ds = r["deaths"]
        R = r["R"]
        nb = [x["t"] for x in ds if x["d_sink"] <= R]
        print(f"{r['method']:22s} seed {r['seed']}  sink nbrs {r['n_sink_nbrs']:2d}  deaths {len(ds):2d}  "
              f"fnd {r['fnd']:.0f}  sink-nbr deaths {len(nb):2d}  first-9 nbr deaths {[round(t) for t in nb[:9]]}  "
              f"alive/connected at end {r['alive_end']}/{r['alive_connected_end']}  "
              f"delivered {r['delivered']} timely {r['timely']}", flush=True)


if __name__ == "__main__":
    main()
