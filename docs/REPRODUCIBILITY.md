# Reproducing the results

Every number, table and data figure of the article is produced from the raw
JSON-lines files in `results/` by the scripts in `analysis/`.  The raw files can
also be regenerated from scratch.  The two schematic figures of the article
(Figures 1 and 2, system model) contain no data and are not part of this
repository.

## 1. Environment

* Python 3.13 (tested with 3.13.18) on Linux; no GPU is needed.
* `pip install -r requirements.txt` (NumPy 2.5.3, SciPy 1.18.1 with HiGHS,
  pandas 3.0.5, Matplotlib 3.11.2, pytest 9.1.1).

## 2. Determinism

* A run is fully determined by `(method, params, scenario, seed)`.  Separate
  random streams are used for topology/shadowing, per-node traffic, the MAC
  and protocol decisions, so the traffic of a seed is identical for every
  method (common random numbers; `tests/test_simulator.py` checks this).
* The batch runner (`experiments/batch.py`) keys each job by a hash of
  `(method, params, scenario, seed, tag)` and skips jobs already present in
  the output file, so an interrupted batch can simply be restarted.
* Floating-point results are bit-identical on the same platform and library
  versions for every method except TwinLoop v1.  TwinLoop v1 solves its
  facility-location MILP with a 2 s *wall-clock* limit (HiGHS), so on a
  loaded machine the solver can stop at a different incumbent and the run can
  differ slightly.  Observed: re-running the E4 configuration of seed 101 gave
  a first-node-death time of 107 s and 100 s instead of the recorded 130 s
  (second repetition: 34 995 instead of 34 919 delivered readings); seeds 102 and 103 reproduced exactly, as did
  every GUARD, BCP and C-ETX run (`analysis/lifetime_deaths.py`).  On other
  platforms the MILP may also return a different optimum when several exist.
  Summary statistics should agree within their confidence intervals.
* Simulation speed (not correctness): after node deaths, a GUARD node without
  a route re-polls at the last hold interval (as small as 0.1 ms), which makes
  some lifetime runs slow (minutes instead of seconds).  Deaths are permanent,
  so such a node cannot regain a route and the outcome is unaffected.

## 3. Steps and running times (2 CPU cores)

| Step | Command | Output | Time |
|------|---------|--------|------|
| unit tests | `make test` | - | < 10 s |
| simulator checks | `make validate` | `results/validation/`, `output/tables/tab_mac_validation.tex` | < 1 min |
| tuning (VAL seeds) | `make tune` | `results/tuning/`, `results/tuned_params.json` | ~30 min |
| test experiments (TEST2 seeds) | `make experiments` | `results/test2/E1..E8.jsonl` | ~5 h (E2 with 300 sensors and BCP dominate) |
| diagnostics (seeds 101-103): route timing, mechanism, lifetime | `make diagnostics` | `results/test2/route_timing.json`, `mechanism.json`, `lifetime_deaths.jsonl` | ~40 min |
| numerical checks of the theory | `make theory` | `results/theory/`, `output/figures/fig_hidden_area.*` | < 1 min |
| figures, tables, summary | `make analysis` | `output/figures/`, `output/tables/`, `results/test2/summary_v2.json` | ~1 min |

`results/test2/summary_v2.json` contains the numbers quoted in the text of the
article, together with the diagnostic records listed in `README.md`.  Running
`make analysis` on the stored raw results reproduces the tables of the article
row by row and its data figures pixel by pixel.  The design iterations of GUARD can be re-run with the scripts in `dev/`
(development seeds 3001-3004 only).  Version-1 results are kept unchanged in
`results/v1/`.

## 4. What may not be changed when re-running

The evaluation protocol of version 2 (`docs/PROTOCOL_v2.md`) fixes the seed
splits, the tuning budget, objective and tie rule, the experiments and the
statistical tests.  It and its amendment A1 (dated in the file) were fixed
before any TEST2 simulation.  Anyone extending the study should design
on DEV seeds, tune on VAL seeds only and report every TEST run.  The version-1
TEST seeds (1-20) are retired and must not be used to evaluate GUARD.
