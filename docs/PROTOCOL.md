# Evaluation protocol (frozen before any test-seed run)

This file fixes how TwinLoop and the baselines are tuned and evaluated.  It was
committed to the repository **before** any simulation on the TEST seeds was
executed (see `git log -- docs/PROTOCOL.md`).  Results on the TEST seeds are
reported in full; no run, seed, load level or metric is excluded.

## 1. Data splits (disjoint seed sets, `src/wsncc/config.py`)

| Set   | Seeds        | Used for |
|-------|--------------|----------|
| TRAIN | 9001 - 9010  | offline training of the RL-CC Q-table only |
| VAL   | 2001 - 2010  | every design iteration of TwinLoop, hyper-parameter tuning of all methods |
| TEST  | 1 - 20       | final evaluation, executed once with frozen configurations |

A seed fixes the topology, link shadowing, traffic (identical for all methods:
common random numbers) and the MAC/protocol random streams.

## 2. Hyper-parameter tuning (equal budget)

* Budget: 12 configurations per method = the default/published setting + 11
  configurations drawn uniformly from the method's search space
  (`SEARCH_SPACES`, fixed RNG seed 12345).
* Evaluation of a configuration: VAL seeds 2001-2002 x offered loads
  {0.5, 1.0, 2.0} readings/s/node, T = 200 s (6 runs).
* Objective: mean timely delivery ratio TDR(1 s) (readings delivered within
  1 s / readings generated); ties broken by the delivery ratio (GDR).
* RL-CC: each configuration's Q-table is trained on TRAIN seeds 9001-9003 at
  loads {1, 2}; the frozen table is then evaluated.
* FSFG-like: the genetic tuning of the fuzzy membership breakpoints is the
  random search above (same budget).

## 3. Final experiments (TEST seeds, frozen configurations)

| ID | Scenario | Loads (readings/s/node) | Seeds |
|----|----------|-------------------------|-------|
| E1 | 100 nodes, 500x500 m, centre sink, first-order energy, T = 300 s | 0.25, 0.5, 1.0, 1.5, 2.0, 3.0 | 1-15 |
| E2 | scalability: N = 50, 100, 200, 300 at constant density | 1.0 | 1-8 |
| E3 | out-of-distribution: corner sink; clustered (Thomas) deployment; critical event bursts; CC2420 state-based energy; harsher interference (Delta = 1.0) | 1.0 (events: 0.5 background) | 1-8 |
| E4 | lifetime: E0 = 0.3 J, T = 900 s | 1.0 | 1-6 |
| E5 | ablations of TwinLoop (E1 scenario) | 0.5, 1.0, 2.0 | 1-10 |
| E6 | sensitivity of TwinLoop to eta, theta0, tau_max (E1 scenario) | 2.0 | 1-8 |
| E7 | clustering solver study: MILP vs greedy on E1 snapshots | 1.0, 2.0 | 1-5 |

## 4. Metrics

GDR (delivered / generated), TDR(1 s) and TDR(2 s), PDR of admitted readings,
mean / median / 95th-percentile end-to-end delay, throughput (readings/s and
kbit/s of payload), energy per delivered reading, remaining energy over time,
first-node-dead and half-nodes-dead times, Jain fairness index of per-source
delivery ratios, link congestion factor (delivered frames / link capacity)
distribution, frames per delivered reading, control overhead, node-side
decision time and sink-side clustering time.  Metrics use readings generated
in [30 s, T - 10 s].

## 5. Statistical analysis

* Primary endpoints: TDR(1 s) and GDR in E1.
* For every (load, metric) family: two-sided Wilcoxon signed-rank tests,
  TwinLoop vs each of the 7 baselines, paired by seed; Holm-Bonferroni
  correction within the family; alpha = 0.05.
* Effect sizes: median paired difference with a 95 % bootstrap CI
  (10,000 resamples) and the matched-pairs rank-biserial correlation.
* Means are reported with 95 % t-confidence intervals.
* Friedman test across all methods per load (E1) with average ranks.

## 6. Amendments (made after tuning, before any TEST-seed run)

These changes were committed after `results/tuned_params.json` was produced on
the VAL seeds and **before** any simulation on the TEST seeds.  None of them
uses information from the TEST seeds.

* **A1 - E5 price ablation.**  The validation search selected `a_p = 0` for
  TwinLoop, i.e. the congestion price does not enter the assignment cost of the
  clustering problem in the tuned configuration.  The pre-registered `no_price`
  variant would therefore be identical to `full`.  It is replaced by
  `with_price` (`a_p = 2`, the default), which measures what the price adds.
* **A2 - new experiment E8.**  Default (untuned) configuration of every method,
  loads {0.5, 1.0, 2.0}, TEST seeds 1-8, T = 300 s.  It shows how much each
  method depends on the validation search.  It is a secondary analysis; the
  primary endpoints remain those of E1.
* **A3 - measurement-only code changes.**  After the commit of this protocol,
  while the tuning batch was running, the simulator gained four additional outputs
  (`queue_max`, `sink_coll_frac`, `sink_util`, `coll_frac_all`) and TwinLoop
  gained an optional `compare_solvers` logging mode for E7 (the greedy
  heuristic is run on the same instance after the MILP and its solution is
  discarded).  Neither change alters any decision of any protocol; the 16
  unit tests pass unchanged.  The tuning workers had loaded the earlier code,
  so none of the 576 tuning records contains the new fields.
* **A4 - E4 job builder.**  An awkward expression that already returned the
  intended job list (all E4 runs record time series) was simplified; the jobs
  are unchanged.
