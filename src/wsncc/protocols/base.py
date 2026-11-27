"""Protocol interface shared by TwinLoop and all baselines."""
from __future__ import annotations

import time

from ..clustering import leachc_heads


class Protocol:
    """Base class.  A protocol decides (i) admission of generated readings,
    (ii) the next hop of every frame, (iii) the clustering, and may run
    periodic controllers in `control()` (called once per simulated second).

    Class attributes read by the simulator:
      clustered        -- uses cluster heads (False: flat collection)
      priority         -- class-aware queues (critical readings first)
      lifo             -- LIFO service of routine traffic
      hold             -- re-evaluation delay when next_hop() returns None [s]
      hdr_extra        -- extra network-header bytes carried in data frames
      reroute_on_retry -- re-run next_hop() before each retransmission
    """

    name = "base"
    clustered = True
    priority = False
    lifo = False
    hold = 0.01
    hdr_extra = 0
    reroute_on_retry = True
    defaults: dict = {}

    def __init__(self, **params):
        p = dict(self.defaults)
        p.update(params)
        self.p = p
        if "hold" in p:
            self.hold = float(p["hold"])
        if "lifo" in p:
            self.lifo = bool(p["lifo"])

    # ------------------------------------------------------------ lifecycle
    def setup(self, sim) -> None:
        self.sim = sim
        self._need_recluster = False
        self.init_state()
        if self.clustered:
            self.recluster()
            self.next_round = sim.now + float(self.p.get("Tc", 30.0))

    def init_state(self) -> None:
        pass

    def tick(self, now: float) -> None:
        if self.clustered and (self._need_recluster or now >= self.next_round):
            self.recluster()
            self.next_round = now + float(self.p.get("Tc", 30.0))
            self._need_recluster = False
        self.control(now)

    def control(self, now: float) -> None:
        pass

    def on_death(self, i: int) -> None:
        self.sim.refresh_routes()
        if self.clustered:
            self._need_recluster = True

    # ------------------------------------------------------------ decisions
    def admit(self, i: int, cls: int) -> bool:  # noqa: ARG002
        return True

    def next_hop(self, i: int, kind: int, final: int, k: int = 1):
        raise NotImplementedError

    def on_tx_result(self, i: int, j: int, frame, ok: bool) -> None:
        pass

    def retry_delay(self, i: int, frame) -> float:  # noqa: ARG002
        """Extra waiting time before a MAC retransmission (default none)."""
        return 0.0

    def on_mac_fail(self, i: int, frame, reason: str) -> bool:  # noqa: ARG002
        """Called when the MAC gives up on a frame.  Return True to put its
        readings back in the queue (network-layer retransmission); the default
        drops them, which is the behaviour of every baseline."""
        return False

    # ----------------------------------------------------------- clustering
    def recluster(self) -> None:
        t0 = time.perf_counter()
        heads, ch_of = self.compute_clusters()
        self.sim.t_sink += time.perf_counter() - t0
        self.sim.n_rounds += 1
        self.sim.set_clusters(heads, ch_of)

    def compute_clusters(self):
        sim = self.sim
        return leachc_heads(sim, int(self.p.get("k", 12)), sim.np_rng, sim.scen["p_assign"])
