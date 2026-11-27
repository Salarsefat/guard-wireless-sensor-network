# Design log

## Version 2: GUARD (DEV seeds only)

Every iteration below used DEV seeds 3001-3004 (lifetime checks: 3001-3003),
T = 150 s unless stated, and the simulator with bidirectional ETX.  No VAL seed
was used for design and no TEST2 seed was simulated before `docs/PROTOCOL_v2.md`
was committed.  Raw records: `dev/*.jsonl`; scripts: `dev/*.py`.  Values are
means over the 4 DEV seeds (TDR = TDR(1 s), p95 in s, energy in mJ per delivered
reading).

### D0. Diagnosis of the baselines and TwinLoop v1 (`dev/diag1.py`)

| method | TDR @0.25 / 1 / 2 / 3 | energy @1 / 3 | finding |
|---|---|---|---|
| C-ETX | 0.989 / 0.921 / 0.716 / 0.499 | 0.478 / 0.789 | readings lost to MAC retry exhaustion (869 per run at 1 r/s vs 4 to channel-access failure, 0 to buffers) |
| BCP | 0.193 / 0.401 / 0.521 / 0.552 | 0.326 / 0.300 | 4.0 readings per frame; highest GDR at 3 r/s (0.727) but p95 delay 2.7-27.5 s |
| CBPR-like | 0.993 / 0.913 / 0.596 / 0.395 | 0.596 / 1.153 | best at 0.25 r/s through short, reliable hops (6.4 hops) |
| TwinLoop v1 | 0.986 / 0.931 / 0.800 / 0.622 | 0.417 / 0.493 | best TDR at >= 1 r/s among the baselines |

22-30 % of the frames addressed to the sink were corrupted at 1 r/s and 52-63 %
at 3 r/s.  After a collision the MAC retries within 0-2.24 ms (BE = 3), less than
the 3.3 ms airtime of a full frame, so hidden-terminal pairs collide again: this
is why retries are exhausted.

### D1. Round 1 (`dev/round1.py`): flat tree, retransmission, hidden-terminal metric, batching

| variant | TDR @0.25 / 1 / 2 / 3 | p95 @1 / 3 | energy @1 / 3 |
|---|---|---|---|
| flat ETX tree | 0.993 / 0.927 / 0.737 / 0.523 | 0.031 / 0.086 | 0.431 / 0.704 |
| + network-layer retransmission (NLR) | 1.000 / 0.991 / 0.764 / 0.458 | 0.051 / 0.965 | 0.443 / 1.082 |
| + hidden-terminal (model) metric | 1.000 / 0.997 / 0.975 / 0.915 | 0.043 / 0.542 | 0.383 / 0.363 |
| + batching (price c0 = 1) | 1.000 / 0.996 / 0.989 / 0.981 | 0.385 / 0.465 | 0.342 / 0.239 |

NLR alone hurts at 3 r/s (more retransmissions into a saturated area); with a
collision-aware tree it is the largest single gain.  A non-zero frame price on an
idle channel made the nodes wait at light load (p95 0.26 s at 0.25 r/s).

### D2. Round 2 (`dev/r2.json`): zero idle price, wait caps, measured ETX

| variant | TDR @1 / 2 / 3 | p95 @1 / 3 | energy @1 / 3 |
|---|---|---|---|
| A: model metric + NLR, no batching | 0.998 / 0.979 / 0.907 | 0.040 / 0.579 | 0.373 / 0.351 |
| C: + batching, price kappa = 1, c0 = 0 | 0.998 / 0.992 / 0.979 | 0.074 / 0.311 | 0.363 / 0.253 |
| D: measured ETX + NLR, no batching | 1.000 / 0.996 / 0.956 | 0.040 / 0.486 | 0.372 / 0.338 |

With c0 = 0 nobody waits at 0.25 r/s (identical to no batching).  Measured ETX
(model-free) matched the model-based metric.  Lifetime warning: with the model
metric a single sink child carried most of the traffic (hottest node 0.90 J in
60 s at 2 r/s vs 0.66 J for C-ETX and 0.32 J for measured ETX).

### D3. Round 3 (`dev/r3.json`): measured ETX + batching

| variant | TDR @1 / 2 / 3 | p95 @1 / 3 | energy @1 / 3 |
|---|---|---|---|
| D2: measured ETX (per-attempt EWMA) + NLR | 0.999 / 0.998 / 0.934 | 0.039 / 0.577 | 0.369 / 0.340 |
| G: + batching (kappa = 1) | 1.000 / 0.999 / 0.998 | 0.082 / 0.285 | 0.354 / 0.244 |
| J: G with the model as prior of unmeasured links | 1.000 / 1.000 / 0.999 | 0.080 / 0.279 | 0.351 / 0.237 |

The model prior adds little and needs two-hop state (about 22 extra beacon bytes),
so GUARD uses the measured metric; the prior is kept as an ablation variant.

### D4. Lifetime (`dev/life1.py`, `dev/life2.py`, `dev/life3.py`; E0 = 0.3 J, T = 900 s, 1 r/s)

| variant | FND [s] | delivered readings | TDR | energy |
|---|---|---|---|---|
| BCP | 217 | 47 351 | 0.265 | 0.419 |
| C-ETX | 105 | 26 864 | 0.377 | 0.737 |
| TwinLoop v1 | 110 | 32 669 | 0.457 | 0.645 |
| G (no energy terms) | 158 | 33 838 | 0.461 | 0.547 |
| + energy balance 0.5 | 295 | 31 736 | 0.416 | 0.582 |
| + scarcity price cE = 2 | 350 | 38 288 | 0.493 | 0.497 |
| + drain-rate price cR = 2 | 361 | 39 974 | 0.512 | 0.479 |

BCP keeps 4 readings per frame at every node by holding readings for seconds,
which a 1 s deadline does not allow; it delivers more readings in total (most of
them late) although its first node dies earlier.  The drain-rate price raised the
p95 delay at 1 r/s, E0 = 2 J from 0.135 s to 0.365 s, so it is off by default and
left to the tuning search space.

### D5. Frozen default (`src/wsncc/protocols/guard.py`)

Measured ETX (alpha_s = 0.05, hysteresis 0.1, energy balance 0.5); priced dispatch
(h = 0.01, kappa = 1, u0 = 0, cE = 2, cR = 0, beta = 0.25); NLR 2 with a
decorrelation delay U(0, 10 ms); EDF; 2 extra beacon bytes charged.  DEV check
(`dev/guard_check.py`): TDR 1.000 / 0.9998 / 0.9995 / 0.998 at 0.25 / 1 / 2 / 3
r/s; p95 0.022 / 0.101 / 0.191 / 0.307 s; energy 0.826 / 0.362 / 0.269 / 0.245 mJ.

### D6. After the TEST2 runs (post hoc observations; nothing was changed)

The frozen configuration (`results/tuned_params.json`) was evaluated once on
TEST2 seeds.  The following observations were made afterwards.  They are
reported in the paper as post hoc and did **not** lead to any change of GUARD,
its parameters or the baselines:

* E5/E6: the decorrelation delay (b > 0), not selected by the tuning, lowers
  energy at 3 r/s but lengthens the 95th-percentile delay at 2 r/s.
* E6: the slack fraction beta never binds for beta >= 0.25 at 2 r/s (identical
  results for 0.25, 0.5, 0.75); waiting always ends earlier.
* E4 diagnostic (`analysis/lifetime_deaths.py`, seeds 101-103): the routing
  energy-balance term (omega) maximises the time to the first death but makes
  the sink's neighbours die within 9-30 s of each other; omega = 0 delivered
  7.0-8.7 % more timely readings over 900 s on these seeds.  A lifetime-aware
  choice of omega is future work (it would have to be tuned on VAL seeds with
  node deaths, not on the test seeds).
* E7: at 1 r/s, GUARD without holding matches C-ETX's mean delay; at 2 r/s no
  holding cost reaches C-ETX's 95th-percentile delay.

## Version 1: TwinLoop (validation seeds; kept for the record)

All iterations below used VAL seeds 2001-2002 and T = 120 s; no TEST seed was
simulated before the protocol in PROTOCOL.md was committed.

| Version | Change | Observation (VAL) |
|---------|--------|-------------------|
| v0 | DPP forwarding with hop-gradient bias theta (H_i - H_j), admission threshold, energy virtual queue, MILP clustering | lost to C-ETX (TDR 0.80 vs 0.91 at 0.5 r/s): gradient term favoured long lossy links; energy queue throttled relays |
| v1 | excess-ETX penalty theta (ETX_ij + H_j - H_i) instead of the hop gradient | ties C-ETX at low load, still no gain under load |
| v2 | per-frame channel cost (V_f + eta Y_j) ETX_ij with Y driven by channel utilisation | almost no batching: utilisation stays < 0.35 while 40 % of frames into the sink collide (hidden terminals) |
| v3 | Y driven by the measured collision fraction at the receiver | batching where collisions occur; +7 pp TDR at 2 r/s vs C-ETX |
| v4 | theta0 = 2 forwarding offset, theta = 16, 2-byte header | ties C-ETX at <= 1 r/s, +8 pp at 2 r/s, +15 pp at 3 r/s |

Simulator changes made during the same period (applied to **all** methods):
protocol-model interference with capture (Gupta-Kumar, Delta = 0.6) instead of a
fixed 1.5 R interference disc; frame coalescing available to every node;
channel-access failures consume a retry instead of dropping the frame;
loss attribution per reading (duplicates no longer double-counted); FSFG-like
uses admitted (not generated) arrivals; BCP and CBPR-like weights multiplied
by link quality.
