# Baselines: what was re-implemented and what was assumed

All nine methods (GUARD and eight baselines) run on the same simulator (`src/wsncc/sim.py`), with the same
topologies, traffic (common random numbers), frame coalescing (up to K = 4
readings per frame), buffers (64 readings), MAC (unslotted IEEE 802.15.4
CSMA/CA) and energy accounting.  They differ only in the `Protocol` subclass in
`src/wsncc/protocols/`.

None of the baselines is the original authors' code: no public implementation
was available to us.  Methods whose published description leaves important
choices open carry the suffix "-like" in the paper.  The table below separates
what we took from the cited article from what we had to assume.  Every assumed
choice was fixed before tuning and every tunable parameter received the same
random-search budget (12 configurations on the validation seeds;
`docs/PROTOCOL.md` for version 1, `docs/PROTOCOL_v2.md` for version 2).

**Version 2 change applied to every baseline.**  A unicast attempt succeeds only
if the data frame (PRR p_ij) and its ACK (p_ji) are received; the simulator has
always drawn both, but in version 1 every method estimated ETX as 1/p_ij.  In
version 2 every link-quality estimate uses q_ij = p_ij * p_ji: the ETX of every
routing tree is 1/q_ij (De Couto et al., 2003), the link-rate estimate R_ij of
BCP and the link weight of CBPR-like are q_ij, and GAN-ACO uses q_ij as its link
quality and q_ij / t_cycle as the capacity in its congestion factor.  Every
method was re-tuned afterwards.  Where the table below says "PRR", read q_ij.

| Method | Source | Taken from the source | Assumed / simplified |
|--------|--------|-----------------------|----------------------|
| **TwinLoop v1** | version 1 of this study (`src/wsncc/protocols/twinloop.py`) | drift-plus-penalty forwarding with excess-ETX bias, collision-driven airtime virtual queue, admission threshold, forced forwarding after tau_max, facility-location MILP clustering (HiGHS) | unchanged except for the ETX definition; re-tuned with its version-1 search space |
| **C-ETX** | LEACH-C clustering (Heinzelman et al., 2002) + minimum-ETX routing (De Couto et al., 2003) | centralised clustering at the sink with k centroids; heads among nodes with at least the mean residual energy; ETX = 1/PRR shortest paths | members join the nearest head with PRR >= 0.7; re-clustering every Tc; FIFO; no congestion control |
| **BCP** | Moeller et al., IPSN 2010, pp. 279-290 | flat backpressure, weight (Q_i - Q_j - V * ETX_ij) * R_ij with R_ij the link-rate estimate, LIFO queues, transmit only when the weight is positive | PRR_ij used as the link-rate estimate R_ij; no floating queues (buffers never overflowed in our runs); default V = 4, the smallest loop-free value in pilot runs on validation seeds (V = 2 stays in the search space and is rejected by tuning); 2-byte backlog field |
| **CBPR-like** | Maheswar et al., Wireless Pers. Commun. 118 (2021) 3167-3185 | heads with high residual energy and short distance to the sink; backpressure on queue length | head score E/Emax - d_sink/d_max per k-means partition; next hop restricted to neighbours closer to the sink (loop avoidance); the article's data-aggregation algorithm is represented by the frame coalescing that every method has |
| **GAN-ACO** | Sefati et al., Mathematics 13 (2025) 597 | GAN with 128/64 hidden units, Adam, lr 1e-3, batch 64; head score with weights (1.5, 1.2, 0.8) for energy, distance to the base station and centrality; ACO weights (1.5, 1.2, 0.8, 1.0, 1.0); rho = 0.1; Q = 0.5; links with CF > 0.7 excluded | **A1**: the article does not define the "real" data of the GAN; we use centroid sets of 64 randomly initialised k-means runs; CF measured every second as delivered frames / (PRR / t_cycle); candidates restricted to neighbours closer to the sink |
| **ACSRO-like** | Narawade & Kolekar, Alexandria Eng. J. 57 (2018) 131-145 | adaptive cuckoo search (Levy flights, decreasing step, nest abandonment) for rate adjustment; the goal of keeping the arrival rate at a parent within its service rate is taken from the description in Sefati et al. (2025), as we could not access the full text | fitness = delivered rate - w_l * overload - w_f * unfairness; occupancy gate q_low below which limits are relaxed; rates enforced as admission probabilities at the members; 10 nests, 20 iterations, pa = 0.25 |
| **FSFG-like** | Mazloomi et al., J. Franklin Inst. 361 (2024) 107014 | fuzzy inference for congestion control with membership functions tuned by a genetic algorithm | inputs (buffer occupancy, unserved fraction of admitted arrivals, CCA failure fraction), zero-order Sugeno rule base, admission probability 1 - gain * degree; the GA is replaced by the common random search over the breakpoints and the gain; the FCM and SVR components are not modelled.  The abstract does not state the control action; we could not access the full text |
| **RL-CC** | inspired by DRL-CaCC, Alipio & Bures, Ad Hoc Networks 166 (2025) 103678 | learning-based adaptation of the sending rate from local congestion observations | tabular Q-learning (48 states x 3 actions) instead of a deep network; no caching; shared Q-table trained offline on TRAIN seeds 9001-9003 at loads 1 and 2 readings/s and frozen |

## Behaviour worth knowing when reading the results

* **ACSRO-like and RL-CC reduce to C-ETX in the tuned configurations.**  On
  the validation seeds, the configurations that never throttle sources were
  the best ones for both methods (their validation TDR(1 s) equals that of
  C-ETX with the same k and Tc to all printed digits).  Losses in these
  scenarios are caused mostly by collisions and retransmission exhaustion
  while buffers stay short, so queue-driven rate control either does not
  trigger (ACSRO-like: occupancy below q_low) or only lowers the delivery
  ratio (RL-CC configurations that throttle score lower).  This is the
  behaviour the paper attributes to queue-based congestion signals; it is
  not a crash or a disabled component.  E8 reports the default
  configurations.
* **BCP has high delivery but long delays.**  Backpressure with LIFO queues
  delivers most readings eventually, but many after more than 1 s, which the
  primary endpoint TDR(1 s) counts as late.
* **GAN-ACO** trains a small GAN at the sink every round; its sink-side time
  per round is therefore much larger than that of the other clustering
  methods.  Its ACO routing, which avoids links with CF > 0.7 and prefers
  short, high-PRR, low-backlog next hops, makes it the strongest baseline in
  our validation runs.
