"""Run single simulations and (for RL-CC) offline training."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

from .config import SCENARIO_DEFAULTS, TRAIN_SEEDS
from .protocols import make_protocol
from .protocols.cetx import RLCC
from .sim import Sim

SIM_VERSION = 2   # bump when simulator behaviour changes (invalidates cached Q-tables)
SCEN_KEYS = ("n", "area", "topo", "sink", "energy", "E0", "lam", "T", "events", "buffer", "pack")


def scenario(**overrides) -> dict:
    sc = dict(SCENARIO_DEFAULTS)
    sc.update(overrides)
    return sc


def _cache_dir() -> Path:
    d = Path(os.environ.get("WSNCC_CACHE", Path(__file__).resolve().parents[2] / "results" / "cache"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def train_rlcc(params: dict, scen_overrides: dict | None = None, seeds=None,
               loads=(1.0, 2.0), episodes_per_seed: int = 1) -> list:
    """Offline training of the shared RL-CC Q-table on TRAIN seeds only.
    Results are cached on disk keyed by (params, scenario, seeds, loads)."""
    seeds = list(seeds or TRAIN_SEEDS[:3])
    key = json.dumps({"p": params, "s": scen_overrides or {}, "seeds": seeds, "loads": list(loads),
                      "e": episodes_per_seed, "sim": SIM_VERSION}, sort_keys=True, default=str)
    h = hashlib.sha1(key.encode()).hexdigest()[:16]
    f = _cache_dir() / f"rlcc_q_{h}.json"
    if f.exists():
        return json.loads(f.read_text())
    Q = np.zeros((48, 3))
    for seed in seeds:
        for lam in loads:
            for _ in range(episodes_per_seed):
                p = {k: v for k, v in params.items() if k != "qtable"}
                p["training"] = True
                proto = RLCC(qtable=Q, **p)
                Sim(scenario(**{**(scen_overrides or {}), "lam": lam}), proto, seed).run()
                Q = proto.Q
    f.write_text(json.dumps(Q.tolist()))
    return Q.tolist()


def run_one(method: str, params: dict | None = None, scen: dict | None = None, seed: int = 1,
            timeseries: bool = False, rl_train_scen: dict | None = None) -> dict:
    sc = scenario(**(scen or {}))
    params = dict(params or {})
    if method == "rlcc" and "qtable" not in params:
        train_sc = rl_train_scen if rl_train_scen is not None else {
            k: v for k, v in (scen or {}).items() if k not in ("lam",)}
        params["qtable"] = train_rlcc({k: v for k, v in params.items()}, train_sc)
    proto = make_protocol(method, params)
    sim = Sim(sc, proto, seed)
    res = sim.run()
    out = {"method": method, "seed": seed}
    out.update({k: sc[k] for k in SCEN_KEYS})
    out["params"] = json.dumps({k: v for k, v in params.items() if k != "qtable"}, sort_keys=True)
    out.update(res)
    log = getattr(proto, "solver_log", None)
    if log:
        ts = [x.get("time", math.nan) for x in log]
        out["milp_time_mean"] = float(np.nanmean(ts))
        out["milp_time_max"] = float(np.nanmax(ts))
        st = [x.get("status", -1) for x in log]
        out["milp_opt_frac"] = float(np.mean([s == 0 for s in st]))
        out["milp_fallbacks"] = int(sum(x.get("fallback", 0) for x in log))
        out["heads_per_round"] = float(np.mean([x.get("n_heads", math.nan) for x in log]))
        gl = [x for x in log if "greedy_obj" in x and x.get("obj") == x.get("obj")]
        if gl:
            out["greedy_gap_pct"] = float(np.mean([100.0 * (x["greedy_obj"] - x["obj"]) / abs(x["obj"])
                                                   for x in gl if abs(x["obj"]) > 1e-12]))
            out["greedy_time_mean"] = float(np.mean([x["greedy_time"] for x in gl]))
            out["milp_n_vars_mean"] = float(np.mean([x.get("n_vars", math.nan) for x in gl]))
    if timeseries:
        out["ts"] = sim.timeseries()
    return out
