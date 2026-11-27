# GUARD: hidden-terminal-aware collection with priced, deadline-safe aggregation

Code and data for the article *"GUARD: Hidden-Terminal-Aware Collection with
Priced, Deadline-Safe Aggregation for Timely Data Delivery in Wireless Sensor
Networks"* by Seyed Salar Sefati and Sanda Maiduc (submitted to *Mathematics*,
MDPI; bibliographic details will be added on publication).

The repository contains:

* a packet-level discrete-event simulator of IEEE 802.15.4 collection networks
  (unslotted CSMA/CA with ACKs and retransmissions, protocol interference model,
  frame coalescing, finite buffers, energy accounting);
* GUARD and the eight re-implemented baselines of the article (TwinLoop, C-ETX,
  BCP, CBPR-like, GAN-ACO, ACSRO-like, FSFG-like, RL-CC);
* the equal-budget tuning, the eight test experiments E1-E8, the instrumented
  diagnostics and the numerical checks of the analysis;
* the raw results of every tuning and test run, and the scripts that turn them
  into every table, data figure and number of the article.

## What GUARD does

GUARD is a flat collection protocol (no cluster heads, no optimisation solver,
no training).  Each node runs four mechanisms:

1. **Collision-aware collection tree.**  Each node estimates, per neighbour, the
   probability that a data frame *and* its ACK get through, from its own
   transmission outcomes (exponentially weighted per attempt), and routes on the
   inverse of this probability plus an energy-balance term, with a loop-free
   hysteresis rule (Lemma 1 and Propositions 1-2 of the article).
2. **Priced, deadline-bounded dispatch.**  A node holding `k < K` readings
   transmits only when `k >= m*`, with the closed-form control limit
   `m* = min{m : m(m+1) >= 2 c Lambda / h}` (Theorem 1), where `Lambda` is its
   arrival rate and `c` a frame price that is zero on an idle channel at full
   energy and grows with channel congestion and energy scarcity.  Waiting is
   capped by a fraction of the oldest reading's deadline slack, so that waiting
   alone cannot cause a deadline miss (Theorem 2).
3. **Recovery.**  Up to `N_r` network-layer retransmission rounds after the MAC
   gives up, and an optional random delay before MAC retransmissions
   (Proposition 3; switched off, `b = 0`, in the tuned configuration).
4. **Earliest-deadline-first** service; readings that are already late go last.

## Requirements

* Python 3.13 (results produced with 3.13.18 on Linux); no GPU is needed.
* `pip install -r requirements.txt` (pinned: NumPy 2.5.3, SciPy 1.18.1 with the
  HiGHS solver, pandas 3.0.5, Matplotlib 3.11.2, pytest 9.1.1).
* Optional: GNU make for the shortcuts below (each target is a single Python
  command, listed in the `Makefile`).

## Quick start

```bash
pip install -r requirements.txt
make test      # 18 unit tests (frame sizes, timing, conservation of readings, determinism)
make smoke     # one short simulation per method (40 nodes, 60 s), a few seconds
```

One simulation from Python:

```python
import sys; sys.path.insert(0, "src")
from wsncc.runner import run_one
r = run_one("guard", {}, {"lam": 2.0, "T": 120.0}, seed=3001)   # development seed
print(r["tdr_1"], r["gdr"], r["delay_p95"], r["energy_per_reading_mJ"])
```

## Reproducing the results

### From the stored raw results (about 1 minute)

```bash
make analysis   # writes output/tables/*.tex, output/figures/*.{pdf,png},
                # results/test2/summary_v2.json, paired tests and scorecard
```

This reads `results/test2/E1.jsonl ... E8.jsonl` and the diagnostic records.
Re-running it reproduces the tables and data figures of the article exactly
(checked: identical table rows and pixel-identical figures).

| Article | Generated file |
|---|---|
| Table 2 (parameters of GUARD) | `output/tables/tab_guard_params.tex` |
| Table 3 (simulation parameters) | `experiments/configs/scenario_defaults.json`, `seed_sets.json` |
| Table 5 (E1, TDR within 1 s) | `output/tables/tab_e1tdr.tex` |
| Table 6 (E1, wins/ties/losses per metric) | `output/tables/tab_e1wtl.tex` |
| Table 7 (E1, computation and overhead) | `output/tables/tab_complexity.tex` |
| Table 8 (E3, robustness) | `output/tables/tab_e3tdr.tex` |
| Table 9 (E4, lifetime) | `output/tables/tab_e4.tex` |
| Table 10 (E5, ablation) | `output/tables/tab_e5.tex` |
| Figure 3 / Figure 4 (E1) | `output/figures/fig_e1_delivery.pdf`, `fig_e1_delay_energy.pdf` |
| Figure 5 / Figure 6 (mechanism) | `output/figures/fig_mechanism.pdf`, `fig_tdr_curve.pdf` |
| Figure 7 (E2) | `output/figures/fig_e2_scalability.pdf` |
| Figure 8 (E4) | `output/figures/fig_e4_lifetime.pdf` |
| Figure 9 (E7) | `output/figures/fig_e7_frontier.pdf` |
| Numbers quoted in the text | `results/test2/summary_v2.json`, `mechanism_summary.json`, `lifetime_deaths.log`, `route_timing.json`; `results/theory/theory_checks.json`; `results/validation/mac_validation.json`; `dev/diag1.jsonl` (loss diagnosis) |

Figures 1 and 2 of the article are schematic drawings and contain no data.
`output/tables/per_metric/` holds one table per metric and experiment (every
metric of E1, E2, E3 and E8 for every method), and `output/` also contains
figures that the article does not show (E3, E6, hidden-zone area).

### From scratch (about 6 hours on two cores)

| Step | Command | Output | Time |
|---|---|---|---|
| simulator checks | `make validate` | `results/validation/mac_validation.json` | < 1 min |
| tuning, validation seeds 2001-2002 | `make tune` | `results/tuning/`, `results/tuned_params.json` | ~30 min |
| experiments E1-E8, test seeds | `make experiments` | `results/test2/E1.jsonl ... E8.jsonl` | ~5 h |
| instrumented diagnostics (seeds 101-103) | `make diagnostics` | `route_timing.json`, `mechanism.json`, `lifetime_deaths.jsonl` | ~40 min |
| numerical checks of the analysis | `make theory` | `results/theory/theory_checks.json` | < 1 min |
| tables, figures, summary | `make analysis` | `output/`, `results/test2/summary_v2.json` | ~1 min |

`make all` runs every step.  `WORKERS=4 make experiments` uses four processes.
The batch runner skips runs that are already in its output file, so an
interrupted step can simply be restarted; delete the output file to re-run it.
`make analysis` reads the stored mechanism records; `make diagnostics` re-runs
them.  `lifetime_deaths.py` only adds missing runs, so delete
`results/test2/lifetime_deaths.jsonl` to repeat them.  Computation times (decision
time per forwarding decision, route-update time) are wall-clock times of this
Python simulator and change from machine to machine.

| Experiment | Content | Runs |
|---|---|---|
| E1 | main scenario, 100 nodes, 0.25-3 readings/s per node, seeds 101-115 | 810 |
| E2 | 50-300 sensors at constant density | 360 |
| E3 | corner sink, clustered deployment, event bursts, CC2420 energy model, Delta = 1.0 and 0.3 | 540 |
| E4 | lifetime, E0 = 0.3 J, 900 s | 90 |
| E5 | ablation of GUARD | 360 |
| E6 | one-at-a-time sensitivity of GUARD | 290 |
| E7 | holding cost h (energy-latency frontier) | 140 |
| E8 | default configurations of every method | 270 |

E2-E8 use seeds 101-110.  `experiments/configs/experiments.json` lists the
scenarios, methods and seeds of every experiment; it is written by
`python experiments/export_configs.py` from the code, like the other files in
`experiments/configs/`.

## Evaluation protocol

`docs/PROTOCOL_v2.md` fixes the seed splits, the tuning budget, objective and tie
rule, the experiments and the statistics; its one amendment (A1) was recorded
before any test run.  Seed sets are disjoint: training 9001-9010 (RL-CC only,
9001-9003 used), development 3001-3020 (design of GUARD, 3001-3004 used; see
`docs/DESIGN_LOG.md` and `dev/`), validation 2001-2010 (tuning, 2001-2002 used)
and test 101-130 (101-115 used).  The test seeds of the earlier evaluation of
TwinLoop (1-20, `results/v1/`, `docs/PROTOCOL.md`) are retired because their
results were seen while designing GUARD.  Every method had the same tuning
budget of 12 configurations; `docs/BASELINES.md` lists what each re-implemented
baseline takes from its publication and what had to be assumed.

## Main results (test seeds; details in the article)

E1, GUARD against the best of the eight baselines at each load:

| Load [readings/s per node] | 0.25 | 0.5 | 1 | 1.5 | 2 | 3 |
|---|---|---|---|---|---|---|
| GUARD TDR(1 s) | 1.000 | 1.000 | 1.000 | 0.9999 | 0.9999 | 0.982 |
| Margin over the best baseline [pp], median [95% CI] | 1.1 [0.6, 1.2] | 2.7 [2.0, 3.1] | 6.5 [5.8, 8.8] | 12.2 [9.5, 14.1] | 18.6 [14.5, 21.8] | 31.8 [28.4, 37.8] |
| GUARD (C-ETX) 95th-percentile delay [ms] | 18 (21) | 47 (26) | 94 (37) | 121 (52) | 138 (69) | 278 (105) |
| GUARD (BCP) energy per delivered reading [mJ] | 0.836 (0.873) | 0.522 (0.485) | 0.370 (0.325) | 0.314 (0.284) | 0.286 (0.274) | 0.275 (0.286) |

Trade-offs: C-ETX and the methods built on it deliver the readings they do
deliver faster from 0.5 readings/s on (while losing 2.7-46% of the readings);
BCP needs less energy per reading at moderate load, with multi-second delays;
GUARD's energy balancing postpones the first node death but makes the sink's
neighbours die almost simultaneously.  `results/test2/scorecard.md` counts, for
every objective, the comparisons that GUARD wins, ties and loses.

## Determinism

A run is determined by `(method, parameters, scenario, seed)`; topology,
shadowing and traffic of a seed are identical for every method (common random
numbers).  Results are bit-identical on the same platform and library versions
for every method except TwinLoop, whose MILP is solved by HiGHS with a 2 s
wall-clock limit, so that a loaded machine can return a different incumbent.
See `docs/REPRODUCIBILITY.md`.

## Repository layout

```
src/wsncc/            simulator (sim.py), radio and MAC constants (radio.py), topologies with
                      bidirectional ETX (topology.py), clustering, GAN for GAN-ACO, runner,
                      configuration (config.py: scenario defaults, seed sets, search spaces);
                      protocols/: guard.py, twinloop.py and the seven other baselines
experiments/          tune.py, run_experiments.py (E1-E8), batch.py, validate_mac.py, smoke.py,
                      export_configs.py; configs/ (JSON copy of the configuration)
analysis/             v2.py (statistics, tables, figures, summary), scorecard.py, mechanism.py,
                      lifetime_deaths.py, route_timing.py (instrumented diagnostics),
                      theory_checks.py and fig_theory.py (numerical checks of the analysis)
results/              tuned_params.json, tuning/, test2/ (raw results of E1-E8, diagnostics,
                      paired tests, summary), theory/, validation/, cache/ (RL-CC Q-tables),
                      v1/ (earlier TwinLoop evaluation, unchanged)
dev/                  design iterations of GUARD on development seeds (scripts and raw records)
docs/                 PROTOCOL_v2.md, PROTOCOL.md (earlier version), DESIGN_LOG.md, BASELINES.md,
                      REPRODUCIBILITY.md
output/               generated tables and figures (make analysis)
tests/                unit tests (pytest)
```

## Citation and license

Code: MIT license (`LICENSE`).  If you use this code, please cite the article
(see `CITATION.cff`).
