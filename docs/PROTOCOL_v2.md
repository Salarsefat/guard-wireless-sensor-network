# Evaluation protocol, version 2 (frozen before any TEST2 run)

Version 1 of this study (TwinLoop, `docs/PROTOCOL.md`) was evaluated on TEST seeds
1-20.  Those results were inspected while diagnosing TwinLoop and designing its
replacement, GUARD, so the version-1 test seeds are **retired**: they are never used
to evaluate GUARD.  This file was committed to the repository **before** hyper-
parameter tuning and before any simulation on the TEST2 seeds (see
`git log -- docs/PROTOCOL_v2.md`).  All TEST2 results are reported; no run, seed, load
level, scenario or metric is excluded.

## 1. Changes to the simulator (apply to every method)

1. **Bidirectional ETX.**  A data frame succeeds only if the frame (p_ij) and its
   ACK (p_ji) are both received; the simulator always drew both, but version 1 let
   every method estimate ETX as 1/p_ij.  ETX is now 1/(p_ij p_ji) (De Couto et al.)
   in every routing tree, in the BCP/CBPR link-rate weights, in GAN-ACO's link
   quality and congestion-factor capacity and in TwinLoop v1.  This strengthens the
   baselines: they now avoid links whose ACKs are lossy.
2. **Behaviour-neutral hooks.**  Per-node airtime counters, an EDF service-order
   flag, `retry_delay()` before a MAC retransmission and `on_mac_fail()` when the
   MAC gives up.  Their defaults reproduce the previous behaviour exactly
   (regression check on DEV seeds: max |difference| 5e-11, JSON round-off).
3. RL-CC Q-table cache keys carry a simulator version, so tables trained before
   the ETX correction are never reused.

The MAC (IEEE 802.15.4 unslotted CSMA/CA, 3 retries), the interference model, the
1 s beacons charged to every method, frame coalescing (K = 4), the 64-reading
buffer and the energy models are unchanged.

## 2. Data splits (disjoint seed sets, `src/wsncc/config.py`)

| Set          | Seeds        | Used for |
|--------------|--------------|----------|
| TRAIN        | 9001 - 9010  | offline training of the RL-CC Q-table only |
| DEV          | 3001 - 3020  | design iterations of GUARD; diagnostics of the baselines (docs/DESIGN_LOG.md) |
| VAL          | 2001 - 2010  | hyper-parameter tuning of all nine methods (2001-2002) |
| TEST2        | 101 - 130    | final evaluation of version 2, executed once with frozen configurations |
| RETIRED_TEST | 1 - 20       | version-1 evaluation; never used for GUARD |

## 3. Hyper-parameter tuning (equal budget, VAL seeds only)

* Methods: GUARD (proposed), TwinLoop v1, C-ETX, BCP, CBPR-like, GAN-ACO,
  ACSRO-like, FSFG-like, RL-CC.  Every method is re-tuned because the ETX
  correction changes the simulator.
* Budget: 12 configurations per method = default setting + 11 drawn uniformly from
  the method's search space (`SEARCH_SPACES`, RNG seed 12345).  GUARD's default is
  the configuration chosen on DEV seeds; its search space varies the congestion
  price kappa, the price threshold u0, the slack fraction beta, the energy prices
  cE and cR, the energy-balance weight, the number of network-layer
  retransmissions, the decorrelation delay, the parent hysteresis and the link-
  estimator weight.  The holding-cost weight h is fixed at 0.01 (only the ratio of
  prices to h matters).
* Evaluation of a configuration: VAL seeds 2001-2002 x offered loads
  {0.5, 1.0, 2.0, 3.0} readings/s/node, T = 200 s (8 runs).  Load 3.0 was added for
  all methods so that high-load behaviour is tuned rather than extrapolated.
* Objective: mean timely delivery ratio TDR(1 s).  Configurations within 0.005 of
  the best mean TDR(1 s) are **tied**; among tied configurations the one with the
  lowest mean 95th-percentile delay is selected, then the one with the higher
  delivery ratio.  (Version 1 broke ties by delivery ratio only; with several
  near-identical TDR values the selection would otherwise be decided by noise.)
* RL-CC: each configuration's Q-table is trained on TRAIN seeds 9001-9003 at loads
  {1, 2}; the frozen table is then evaluated.

## 4. Final experiments (TEST2 seeds, frozen configurations)

| ID | Scenario | Loads (readings/s/node) | Seeds |
|----|----------|-------------------------|-------|
| E1 | 100 nodes, 500x500 m, centre sink, first-order energy, T = 300 s | 0.25, 0.5, 1, 1.5, 2, 3 | 101-115 |
| E2 | scalability: N = 50, 100, 200, 300 at constant density | 1.0 | 101-110 |
| E3 | robustness: corner sink; clustered (Thomas) deployment; critical event bursts (0.5 background); CC2420 state-based energy; harsher interference (Delta = 1.0); milder interference (Delta = 0.3) | 1.0 | 101-110 |
| E4 | lifetime: E0 = 0.3 J, T = 900 s, time series | 1.0 | 101-110 |
| E5 | ablation of GUARD: no dispatch, no network-layer retransmission, no decorrelation, no EDF, static ETX, no energy terms, model prior, CTP-like core | 0.5, 1, 2, 3 | 101-110 |
| E6 | one-at-a-time sensitivity of GUARD: kappa, beta, alpha_s, nlr | 2.0 | 101-110 |
| E7 | energy-latency frontier of GUARD: h in {0.0025 ... 0.08} and no holding | 1.0, 2.0 | 101-110 |
| E8 | default (untuned) configuration of every method | 0.5, 1, 2 | 101-110 |

Ten seeds are used for E2-E8 (version 1 used 8): with 8 seeds a Holm-corrected
exact Wilcoxon test over 8 comparisons cannot reach alpha = 0.05 even under perfect
separation (smallest adjusted p = 8 x 2/2^8 = 0.0625); with 10 seeds it can
(0.0156).

## 5. Metrics

TDR(0.5 s), TDR(1 s), TDR(2 s); GDR (delivered / generated); PDR of admitted
readings; throughput; mean, median, 95th and 99th percentile delay of delivered
readings; energy per delivered reading; first-node-dead and half-nodes-dead
times, alive fraction and total readings delivered (E4); Jain fairness index of
per-source delivery ratios; loss counts by cause; fraction of frames corrupted by
interference (all receivers and at the sink); frames per delivered reading and
readings per frame; control bytes; node-side decision time and route/cluster
computation time.  Metrics use readings generated in [30 s, T - 10 s].

Delay statistics are conditional on delivery.  When methods deliver very different
fractions of readings, conditional delay favours the method that loses the hard
readings; the paper therefore also reports TDR(tau) for three deadlines (the
unconditional delay distribution evaluated at tau).

## 6. Statistical analysis

* Primary endpoints: TDR(1 s) and GDR in E1.
* For every (experiment, condition, metric) family: two-sided Wilcoxon signed-rank
  tests of GUARD against each of the 8 baselines, paired by seed;
  Holm-Bonferroni correction over the 8 comparisons; alpha = 0.05.  Effect sizes:
  median paired difference with a 95 % bootstrap confidence interval (10 000
  resamples) and the matched-pairs rank-biserial correlation.
* Friedman test over all methods per E1 load.
* "GUARD is better than X on metric M" is claimed only when the Holm-adjusted
  p-value is below 0.05 and the direction of the median difference favours GUARD.
  Every comparison, favourable or not, is tabulated.

## 7. Amendments

Any change after this file is committed is recorded here with its date and
reason, before the affected runs.

**A1 (2026-11-26, after GUARD's tuning runs on VAL, before any TEST2 run).**
The tuning selected GUARD configuration 7 (all 12 GUARD configurations were
within 0.3 pp of the best validation TDR(1 s), so the pre-registered tie rule
chose the lowest 95th-percentile delay).  It has decorrelation delay b = 0 and
price threshold u0 = 0.3.  Consequences:
* E5: the pre-registered variant `no_decorrelation` (b = 0) would be identical
  to the tuned method.  It is replaced by `with_decorrelation` (b = 10 ms, the
  default), which measures the effect of adding the delay.
* E6: one-at-a-time sweeps of b in {0, 5, 10, 20, 50} ms and of u0 in
  {0, 0.1, 0.2, 0.3, 0.4} are added, because both tuned values lie at the edge of
  their search ranges.
No other experiment changes.
