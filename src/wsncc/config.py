"""Scenario defaults, seed sets and hyper-parameter search spaces.

The seed sets are disjoint (docs/PROTOCOL_v2.md):
  * TRAIN seeds  -- only used to train the RL-CC Q-table;
  * DEV seeds    -- design iterations of GUARD (and diagnostics of the baselines);
  * VAL seeds    -- hyper-parameter tuning of every method (equal budget);
  * TEST2 seeds  -- final held-out evaluation of version 2, run once with frozen settings;
  * RETIRED_TEST -- the version-1 test seeds; their results were seen while designing
                    GUARD and are therefore never used to evaluate it.
"""
from __future__ import annotations

SCENARIO_DEFAULTS: dict = {
    # deployment
    "n": 100, "area": 500.0, "R": 100.0, "topo": "uniform", "sink": "center",
    # channel
    "intf_factor": 1.5, "capture_delta": 0.6, "prr_width": 0.06, "shadow_db": 2.0, "p_route": 0.5, "p_assign": 0.7,
    # energy / buffers / framing
    "energy": "first_order", "E0": 2.0, "buffer": 64, "pack": 4,
    # traffic and timing
    "lam": 1.0, "T": 300.0, "warm": 30.0, "drain": 10.0,
    # critical event bursts (disabled by default)
    "events": False, "ev_rate": 0.05, "ev_radius": 60.0, "ev_dur": 10.0, "ev_lam": 4.0,
}

TRAIN_SEEDS = list(range(9001, 9011))
DEV_SEEDS = list(range(3001, 3021))
VAL_SEEDS = list(range(2001, 2011))
TEST2_SEEDS = list(range(101, 131))
RETIRED_TEST_SEEDS = list(range(1, 21))
TEST_SEEDS = TEST2_SEEDS          # the evaluation seeds of the current version

METHODS = ["guard", "twinloop", "cetx", "bcp", "cbpr", "ganaco", "acsro", "fsfg", "rlcc"]
BASELINES = METHODS[1:]

# Search spaces for the equal-budget random search (each method: 12 configurations,
# the first being its published/default setting).
SEARCH_SPACES: dict = {
    "guard": {"kappa": [0.5, 1.0, 2.0], "u0": [0.0, 0.2, 0.3], "beta": [0.1, 0.25, 0.5],
              "cE": [0.0, 1.0, 2.0], "cR": [0.0, 0.5], "ebal": [0.0, 0.5, 1.0],
              "nlr": [1, 2, 3], "rdel": [0.0, 0.01, 0.03], "hyst": [0.05, 0.1, 0.2],
              "alpha_s": [0.05, 0.1]},
    "twinloop": {"theta": [8.0, 16.0, 32.0], "theta0": [1.0, 2.0, 4.0], "eta": [5.0, 10.0, 20.0],
                 "phi_target": [0.10, 0.15, 0.20], "Vf": [0.25, 0.5, 1.0],
                 "tau_max": [0.25, 0.5, 1.0], "Va": [24.0, 48.0], "f0": [1.5, 3.0, 6.0],
                 "a_p": [0.0, 1.0, 2.0, 4.0], "kappa": [0.4, 0.6, 0.8], "Tc": [20.0, 30.0, 60.0],
                 "lifo": [False, True]},
    "cetx": {"k": [8, 10, 12, 15, 20, 25], "Tc": [20.0, 30.0, 60.0]},
    "bcp": {"V": [2.0, 4.0, 6.0, 8.0, 12.0, 16.0], "hold": [0.005, 0.01, 0.02]},
    "cbpr": {"k": [8, 10, 12, 15, 20, 25], "Tc": [20.0, 30.0, 60.0], "hold": [0.005, 0.01, 0.02]},
    "ganaco": {"k": [8, 10, 12, 15, 20, 25], "Tc": [20.0, 30.0, 60.0], "rho": [0.05, 0.1, 0.2],
               "Q": [0.25, 0.5, 1.0], "cf_thr": [0.6, 0.7, 0.8]},
    "acsro": {"k": [8, 10, 12, 15, 20, 25], "Tc": [20.0, 30.0, 60.0], "Tctrl": [1.0, 2.0, 5.0],
              "w_l": [1.0, 2.0, 4.0, 8.0], "q_low": [0.1, 0.25, 0.4]},
    "fsfg": {"k": [8, 10, 12, 15, 20, 25], "Tc": [20.0, 30.0, 60.0], "Tctrl": [1.0, 2.0, 5.0],
             "c_buf": [0.3, 0.5, 0.7], "c_cf": [0.3, 0.5, 0.7], "c_cca": [0.3, 0.5, 0.7],
             "gain": [0.5, 0.75, 1.0]},
    "rlcc": {"k": [8, 10, 12, 15, 20, 25], "Tc": [20.0, 30.0, 60.0], "alpha": [0.05, 0.1, 0.2],
             "gamma": [0.8, 0.9], "w_l": [0.5, 1.0, 2.0], "w_b": [0.5, 1.0, 2.0]},
}

TUNING_BUDGET = 12
TUNE_LOADS = [0.5, 1.0, 2.0, 3.0]
TUNE_T = 200.0
TIE_TOL = 0.005        # configurations within 0.5 pp of the best mean TDR(1 s) are tied
