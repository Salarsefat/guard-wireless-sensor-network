"""Discrete-event, packet-level simulator of a cluster-based WSN.

PHY/MAC: IEEE 802.15.4 2.4 GHz, non-beacon (unslotted) CSMA/CA with ACKs and
retransmissions (macMinBE = 3, macMaxBE = 5, macMaxCSMABackoffs = 4,
macMaxFrameRetries = 3).  Interference follows the protocol model of Gupta and
Kumar with guard factor Delta: a frame s -> r is corrupted if, at any time
during the frame, another transmitter t with d(t, r) < (1 + Delta) d(s, r) is
active, if r transmits (half duplex) or if r is already receiving another
frame.  Carrier sense detects any transmitter within cs_factor * R.  ACK frames
do not occupy the channel (0.35 ms); an ACK is lost with the complementary PRR
of the reverse link.

Network layer: every queued reading is stored individually (buffer capacity is
counted in readings).  When a node gets a transmission opportunity it packs up
to `pack` readings of the same direction into one frame (frame coalescing --
available to every protocol).  Members send their own readings to their cluster
head (CH) in COLLECT frames; CHs and relays send UPSTREAM frames towards the
sink.  Admission, next-hop selection and clustering are delegated to a
Protocol object (protocols/base.py).

A reading is the list [rid, origin, t_gen, cls, hops, t_enq].
"""
from __future__ import annotations

import heapq
import itertools
import math
import random
import time
from collections import deque

import numpy as np

from . import radio as R
from .topology import Topology, build_topology, etx_distances

# event types
EV_GEN, EV_BACKOFF, EV_TXSTART, EV_TXEND, EV_ACK, EV_TRY, EV_TICK, EV_GENC = range(8)
# MAC states
IDLE, CSMA, TX, WAIT, HOLD, DEAD = range(6)
# frame kinds / roles
COLLECT, UPSTREAM = 0, 1
MEMBER, HEAD = 0, 1
TTL = 40
DROP_REASONS = ("buf", "caf", "retry", "ttl", "dead")
# reading fields
RID, ORIG, TGEN, CLS, HOPS, TENQ = range(6)


class Frame:
    __slots__ = ("fid", "kind", "src", "dst", "final", "readings", "bits",
                 "retries", "nb", "be", "corrupt")


class Sim:
    """One simulation run.  `scen` is a dict of scenario parameters (see
    config.SCENARIO_DEFAULTS); `proto` a Protocol instance."""

    def __init__(self, scen: dict, proto, seed: int, topology: Topology | None = None):
        self.scen = scen
        self.seed = seed
        self.proto = proto
        n = scen["n"]
        self.n = n
        self.N = N = n + 1
        self.sink = n
        self.top = topology or build_topology(
            n, scen["area"], scen["R"], kind=scen["topo"], sink=scen["sink"],
            intf_factor=scen["intf_factor"], prr_width=scen["prr_width"],
            shadow_db=scen["shadow_db"], p_route=scen["p_route"], seed=seed)
        top = self.top
        self.pos = top.pos
        self.nbr = top.nbr
        self.prr = top.prr
        self.lq = top.lq
        self.dist = top.dist
        self.intf = top.intf
        self.energy = R.make_energy_model(scen["energy"])
        self.B = scen["buffer"]
        self.T = scen["T"]
        self.warm = scen["warm"]
        self.t_stop = scen["T"] - scen["drain"]
        self.lam = scen["lam"]

        # protocol behavioural flags
        self.priority = bool(getattr(proto, "priority", False))
        self.lifo = bool(getattr(proto, "lifo", False))
        self.hdr_extra = int(getattr(proto, "hdr_extra", 0))
        self.pack = min(scen["pack"], R.max_pack(self.hdr_extra))
        self.full_bits = R.frame_bits(self.pack, self.hdr_extra)
        self.t_cycle = R.service_cycle(self.full_bits)

        # random streams (traffic is method-independent: common random numbers)
        self.rng = random.Random(seed * 7919 + 1)            # MAC / channel
        self.rng_node = [random.Random(seed * 104729 + 17 * i + 3) for i in range(n)]
        self.prng = random.Random(seed * 31 + 7)             # protocol decisions
        self.np_rng = np.random.default_rng(seed * 31 + 7)

        # node state
        self.alive = [True] * N
        self.E = [scen["E0"]] * N
        self.E[self.sink] = math.inf
        self.used = [0.0] * N
        self.role = [HEAD] * N
        self.ch = list(range(N))
        self.H = [0.0] * N
        self.parent = [-1] * N
        self.adv_q = [0] * N          # last advertised backlog (piggy-back / beacon)
        self.adv_z = [0.0] * N        # last advertised energy virtual queue (TwinLoop)
        self.qlen = [0] * N           # readings held by the node (queues + MAC frame)
        self.nq = [0] * N             # readings in queues (excl. MAC frame) -> buffer use
        self.pend = [deque() for _ in range(N)]   # member's own readings (COLLECT)
        self.up = [deque() for _ in range(N)]     # upstream readings
        self.cpend = [deque() for _ in range(N)]
        self.cup = [deque() for _ in range(N)]
        self.mac = [IDLE] * N
        self.macf: list[Frame | None] = [None] * N
        self.tok = [0] * N
        self.txing: list[Frame | None] = [None] * N
        self.rxf: list[Frame | None] = [None] * N
        self.active: list[Frame] = []          # frames currently on the air
        self.gk = 1.0 + float(scen.get("capture_delta", 1.0))
        self.busy = [0] * N
        self.busy_since = [0.0] * N
        self.busy_acc = [0.0] * N
        self.lastfid = [dict() for _ in range(N)]
        # per-node counters usable by controllers
        self.cnt_cca = [0] * N
        self.cnt_ccafail = [0] * N
        self.cnt_txok = [0] * N
        self.cnt_txfail = [0] * N
        self.cnt_out = [0] * N        # readings successfully forwarded
        self.cnt_in = [0] * N         # readings received from other nodes
        self.cnt_in_collect = [0] * N  # of which in COLLECT frames (head role)
        self.cnt_gen = [0] * N        # readings generated
        self.cnt_adm = [0] * N        # readings admitted
        self.cnt_drop = [0] * N       # readings dropped at this node (buffer)
        self.cnt_rx_att = [0] * N     # frames addressed to this node
        self.cnt_rx_coll = [0] * N    # ... of which corrupted by interference
        self.cnt_air = [0] * N        # frames put on the air by this node
        self.air_s = [0.0] * N        # seconds of airtime used by this node
        self.edf = bool(getattr(proto, "edf", False))
        self.edf_tau = float(getattr(proto, "edf_tau", math.inf))

        # metrics
        self.now = 0.0
        self.heap: list = []
        self._seq = itertools.count()
        self._fid = itertools.count(1)
        self._rid = itertools.count(1)
        self.gen_win = [0, 0]
        self.adm_win = [0, 0]
        self.sup_win = [0, 0]
        self.del_win = [0, 0]
        self.drop_reason: dict[int, str] = {}     # last loss reason of window readings
        self.drop_buf_head = 0
        self.dup_at_sink = 0
        self.qmax = 0
        self.gen_src = [0] * N
        self.del_src = [0] * N
        self.delays: list[float] = []
        self.delays_c: list[float] = []
        self.hops_del: list[int] = []
        self.delivered_ids: set[int] = set()
        self.del_total = 0
        self.del_sec = 0
        self.link_ok: dict[int, int] = {}
        self.tx_frames = 0
        self.tx_readings = 0
        self.ctrl_bytes = 0
        self.data_bytes = 0
        self.t_dec = 0.0
        self.n_dec = 0
        self.t_sink = 0.0
        self.n_rounds = 0
        self.alive_count = n
        self.fnd = None
        self.hnd = None
        self.ts_t: list[float] = []
        self.ts_alive: list[int] = []
        self.ts_energy: list[float] = []
        self.ts_deliv: list[int] = []
        self.ts_q: list[float] = []
        self.ts_heads: list[int] = []

    # ------------------------------------------------------------------ utils
    def schedule(self, t: float, et: int, i: int, arg=None) -> None:
        heapq.heappush(self.heap, (t, next(self._seq), et, i, arg))

    def charge(self, i: int, e: float) -> None:
        if i == self.sink or e <= 0.0:
            return
        self.E[i] -= e
        self.used[i] += e
        if self.E[i] <= 0.0 and self.alive[i]:
            self.kill(i)

    def alive_nbrs(self, i: int) -> list[int]:
        al = self.alive
        return [j for j in self.nbr[i] if al[j]]

    def refresh_routes(self) -> None:
        self.H, self.parent = etx_distances(self.top, self.alive)

    def set_clusters(self, heads: list[int], ch_of) -> None:
        """Install a clustering (called by protocols) and charge its control cost:
        one control frame per alive node (dissemination) received by its neighbours."""
        role = self.role
        ch = self.ch
        for i in range(self.n):
            role[i] = MEMBER
            ch[i] = i
        hs = set(heads)
        items = ch_of.items() if isinstance(ch_of, dict) else enumerate(ch_of)
        for i, h in items:
            if i < self.n and h != i and i not in hs:
                ch[i] = h
        for i in range(self.n):
            if ch[i] == i:            # heads and unassigned nodes act as their own head
                role[i] = HEAD
                # a new head's pending own readings become upstream readings
                if self.pend[i] or self.cpend[i]:
                    self.up[i].extend(self.pend[i])
                    self.pend[i].clear()
                    self.cup[i].extend(self.cpend[i])
                    self.cpend[i].clear()
        bits = (R.PHY_OVERHEAD + R.MAC_OVERHEAD + 8) * 8
        en = self.energy
        etx = en.tx(bits, self.top.R)
        erx = en.rx(bits)
        for i in range(self.n):
            if self.alive[i]:
                k = 0
                for j in self.nbr[i]:
                    if self.alive[j]:
                        k += 1
                self.charge(i, etx + erx * k)
                self.ctrl_bytes += bits // 8

    def head_age(self, i: int) -> float:
        """Age of the oldest queued upstream reading (0 if none)."""
        a = 0.0
        for dq in (self.cup[i], self.up[i]):
            if dq:
                a = max(a, self.now - dq[0][TENQ])
        return a

    # --------------------------------------------------------------- dropping
    def _drop(self, readings, reason: str, node: int = -1) -> None:
        w0, w1 = self.warm, self.t_stop
        dr = self.drop_reason
        for rd in readings:
            if w0 <= rd[TGEN] < w1:
                dr[rd[RID]] = reason
        if reason == "buf" and node >= 0:
            self.cnt_drop[node] += len(readings)
            if self.role[node] == HEAD:
                self.drop_buf_head += len(readings)

    def kill(self, i: int) -> None:
        if not self.alive[i]:
            return
        self.alive[i] = False
        self.mac[i] = DEAD
        self.tok[i] += 1
        lost = []
        for dq in (self.pend[i], self.cpend[i], self.up[i], self.cup[i]):
            lost.extend(dq)
            dq.clear()
        f = self.macf[i]
        if f is not None:
            lost.extend(f.readings)
            self.macf[i] = None
        self._drop(lost, "dead")
        self.qlen[i] = 0
        self.nq[i] = 0
        self.adv_q[i] = 0
        self.alive_count -= 1
        if self.fnd is None:
            self.fnd = self.now
        if self.hnd is None and self.alive_count <= self.n / 2:
            self.hnd = self.now
        self.proto.on_death(i)

    # ---------------------------------------------------------------- queues
    def _accept(self, i: int, readings, own: bool) -> None:
        """Store readings at node i (buffer counted in readings)."""
        free = self.B - self.nq[i]
        k = len(readings)
        if k > free:
            # critical readings may evict routine ones (priority protocols only)
            if self.priority:
                crit = [rd for rd in readings if rd[CLS] == 1]
                while len(crit) > self.B - self.nq[i] and self._evict_routine(i):
                    pass
                free = self.B - self.nq[i]
                keep = crit[:max(free, 0)]
                rest_free = free - len(keep)
                rout = [rd for rd in readings if rd[CLS] != 1]
                keep += rout[:max(rest_free, 0)]
                kset = set(id(r) for r in keep)
                self._drop([rd for rd in readings if id(rd) not in kset], "buf", i)
                readings = keep
            else:
                self._drop(readings[max(free, 0):], "buf", i)
                readings = readings[:max(free, 0)]
            if not readings:
                return
        now = self.now
        member = own and self.role[i] == MEMBER and self.ch[i] != i
        pr = self.priority
        for rd in readings:
            rd[TENQ] = now
            if pr and rd[CLS] == 1:
                (self.cpend if member else self.cup)[i].append(rd)
            else:
                (self.pend if member else self.up)[i].append(rd)
        k = len(readings)
        self.nq[i] += k
        self.qlen[i] += k
        if self.qlen[i] > self.qmax:
            self.qmax = self.qlen[i]
        st = self.mac[i]
        if st == IDLE or st == HOLD:
            self._try_send(i)

    def _evict_routine(self, i: int) -> bool:
        cands = [dq for dq in (self.up[i], self.pend[i]) if dq]
        if not cands:
            return False
        dq = max(cands, key=lambda d: d[-1][TENQ])
        rd = dq.pop()
        self.nq[i] -= 1
        self.qlen[i] -= 1
        self._drop((rd,), "buf", i)
        return True

    def _head_source(self, i: int) -> int:
        """Which queue is served next: 0 pend, 1 up, 2 cpend, 3 cup, -1 empty."""
        if self.priority:
            cp, cu = self.cpend[i], self.cup[i]
            if cp or cu:
                if cp and cu:
                    return 2 if cp[0][TENQ] <= cu[0][TENQ] else 3
                return 2 if cp else 3
        p, u = self.pend[i], self.up[i]
        if p and u:
            if self.lifo:
                return 0 if p[-1][TENQ] >= u[-1][TENQ] else 1
            return 0 if p[0][TENQ] <= u[0][TENQ] else 1
        if p:
            return 0
        if u:
            return 1
        return -1

    # ------------------------------------------------------------------- MAC
    def _try_send(self, i: int) -> None:
        st = self.mac[i]
        if st != IDLE and st != HOLD:
            return
        src = self._head_source(i)
        if src < 0:
            self.mac[i] = IDLE
            return
        dq = (self.pend, self.up, self.cpend, self.cup)[src][i]
        if (src == 0 or src == 2) and self.role[i] == MEMBER and self.ch[i] != i \
                and self.alive[self.ch[i]]:
            kind, final = COLLECT, self.ch[i]
        else:
            kind, final = UPSTREAM, self.sink
        k = min(self.pack, len(dq))
        t0 = time.perf_counter()
        nh = self.proto.next_hop(i, kind, final, k)
        self.t_dec += time.perf_counter() - t0
        self.n_dec += 1
        if nh is None or nh < 0 or not self.alive[nh]:
            self.mac[i] = HOLD
            self.tok[i] += 1
            self.schedule(self.now + self.proto.hold, EV_TRY, i, self.tok[i])
            return
        if self.edf and len(dq) > 1:
            # earliest deadline first; readings already past the deadline go last
            lim = self.now - self.edf_tau
            items = sorted(dq, key=lambda rd: (rd[TGEN] < lim, rd[TGEN]))
            dq.clear()
            dq.extend(items)
            readings = [dq.popleft() for _ in range(k)]
        elif src == 1 and self.lifo:
            readings = [dq.pop() for _ in range(k)]
        else:
            readings = [dq.popleft() for _ in range(k)]
        self.nq[i] -= k
        f = Frame()
        f.fid = next(self._fid)
        f.kind = kind
        f.src = i
        f.dst = nh
        f.final = final
        f.readings = readings
        f.bits = R.frame_bits(k, self.hdr_extra)
        f.retries = 0
        f.nb = 0
        f.be = R.MAC_MIN_BE
        f.corrupt = False
        self.macf[i] = f
        self.mac[i] = CSMA
        self.tok[i] += 1
        self.schedule(self.now + self.rng.randrange(1 << f.be) * R.UNIT_BACKOFF,
                      EV_BACKOFF, i, self.tok[i])

    def _requeue(self, i: int, f: Frame) -> None:
        """Return a frame's readings to the queue head and hold."""
        member = f.kind == COLLECT
        for rd in reversed(f.readings):
            crit = self.priority and rd[CLS] == 1
            if member:
                dq = (self.cpend if crit else self.pend)[i]
            else:
                dq = (self.cup if crit else self.up)[i]
            if self.lifo and not crit:
                dq.append(rd)
            else:
                dq.appendleft(rd)
        self.nq[i] += len(f.readings)
        self.macf[i] = None
        self.mac[i] = HOLD
        self.tok[i] += 1
        self.schedule(self.now + self.proto.hold, EV_TRY, i, self.tok[i])

    def _release(self, i: int, f: Frame, reason: str | None) -> None:
        self.qlen[i] -= len(f.readings)
        if reason is not None:
            self._drop(f.readings, reason)
        self.macf[i] = None
        self.mac[i] = IDLE
        self._try_send(i)

    def _backoff_end(self, i: int, token: int) -> None:
        if token != self.tok[i] or not self.alive[i]:
            return
        f = self.macf[i]
        self.cnt_cca[i] += 1
        self.charge(i, self.energy.cca)
        if not self.alive[i]:
            return
        if self.busy[i] == 0:
            self.schedule(self.now + R.T_CCA + R.T_TURN, EV_TXSTART, i, token)
            return
        self.cnt_ccafail[i] += 1
        f.nb += 1
        if f.nb > R.MAC_MAX_CSMA_BACKOFFS:
            # channel-access failure: the network layer retries the frame with a new
            # CSMA procedure; this consumes one of the macMaxFrameRetries attempts
            self.cnt_txfail[i] += 1
            f.retries += 1
            if f.retries > R.MAC_MAX_FRAME_RETRIES:
                if self.proto.on_mac_fail(i, f, "caf"):
                    self._requeue(i, f)
                else:
                    self._release(i, f, "caf")
                return
            f.nb = 0
            f.be = R.MAC_MIN_BE
            self.schedule(self.now + self.rng.randrange(1 << f.be) * R.UNIT_BACKOFF,
                          EV_BACKOFF, i, token)
            return
        f.be = min(f.be + 1, R.MAC_MAX_BE)
        self.schedule(self.now + self.rng.randrange(1 << f.be) * R.UNIT_BACKOFF,
                      EV_BACKOFF, i, token)

    def _tx_start(self, i: int, token: int) -> None:
        if token != self.tok[i] or not self.alive[i]:
            return
        f = self.macf[i]
        r = f.dst
        self.charge(i, self.energy.tx(f.bits, self.dist[i][r]))
        if not self.alive[i]:
            return
        # piggy-backed advertisement: backlog that remains if this frame succeeds
        self.adv_q[i] = self.qlen[i] - len(f.readings)
        self.txing[i] = f
        busy = self.busy
        rxf = self.rxf
        now = self.now
        bs = self.busy_since
        for u in self.intf[i]:          # carrier-sense neighbourhood
            if busy[u] == 0:
                bs[u] = now
            busy[u] += 1
        pos = self.pos
        gk = self.gk
        xi, yi = pos[i]
        xr, yr = pos[r]
        d_ir = self.dist[i][r]
        hyp = math.hypot
        for g in self.active:           # Gupta-Kumar protocol model
            u = g.dst
            if not g.corrupt:
                xu, yu = pos[u]
                if hyp(xi - xu, yi - yu) < gk * self.dist[g.src][u]:
                    g.corrupt = True
            if not f.corrupt:
                xt, yt = pos[g.src]
                if hyp(xt - xr, yt - yr) < gk * d_ir:
                    f.corrupt = True
        if self.txing[r] is not None or rxf[r] is not None:
            f.corrupt = True
        else:
            rxf[r] = f
        g = rxf[i]
        if g is not None:                # half duplex: i stops receiving
            g.corrupt = True
        self.active.append(f)
        self.mac[i] = TX
        self.cnt_air[i] += 1
        self.air_s[i] += f.bits / R.BITRATE
        self.tx_frames += 1
        self.tx_readings += len(f.readings)
        self.data_bytes += f.bits // 8
        self.schedule(now + f.bits / R.BITRATE, EV_TXEND, i, f)

    def _tx_end(self, i: int, f: Frame) -> None:
        now = self.now
        busy = self.busy
        bs = self.busy_since
        acc = self.busy_acc
        for u in self.intf[i]:
            busy[u] -= 1
            if busy[u] == 0:
                acc[u] += now - bs[u]
        self.txing[i] = None
        self.active.remove(f)
        r = f.dst
        if self.rxf[r] is f:
            self.rxf[r] = None
        self.cnt_rx_att[r] += 1
        if f.corrupt:
            self.cnt_rx_coll[r] += 1
        if not self.alive[i]:
            return
        en = self.energy
        ok = False
        if self.alive[r]:
            if r != self.sink:
                self.charge(r, en.rx(f.bits))
            if self.alive[r] and not f.corrupt and self.rng.random() < self.prr[i][r]:
                ok = True
        if ok:
            lf = self.lastfid[r]
            if lf.get(i) != f.fid:
                lf[i] = f.fid
                self._receive(r, f)
            # the ACK piggy-backs the receiver's updated backlog
            if r != self.sink:
                self.adv_q[r] = self.qlen[r]
            ack_ok = self.rng.random() < self.prr[r][i]
            if r != self.sink:
                self.charge(r, en.tx(R.ACK_BITS, self.dist[i][r]))
            if ack_ok:
                self.charge(i, en.rx(R.ACK_BITS))
                t_res = now + R.T_TURN + R.T_ACK
            else:
                self.charge(i, en.ack_listen)
                t_res = now + R.T_ACKWAIT
        else:
            ack_ok = False
            self.charge(i, en.ack_listen)
            t_res = now + R.T_ACKWAIT
        if not self.alive[i]:
            return
        self.mac[i] = WAIT
        self.schedule(t_res, EV_ACK, i, (f, ack_ok))

    def _ack(self, i: int, f: Frame, ok: bool) -> None:
        if not self.alive[i] or self.macf[i] is not f:
            return
        self.proto.on_tx_result(i, f.dst, f, ok)
        if ok:
            self.cnt_txok[i] += 1
            self.cnt_out[i] += len(f.readings)
            if self.warm <= self.now < self.t_stop:
                key = i * self.N + f.dst
                self.link_ok[key] = self.link_ok.get(key, 0) + 1
            self._release(i, f, None)
            return
        self.cnt_txfail[i] += 1
        f.retries += 1
        if f.retries > R.MAC_MAX_FRAME_RETRIES:
            if self.proto.on_mac_fail(i, f, "retry"):
                self._requeue(i, f)      # network-layer retransmission
            else:
                self._release(i, f, "retry")
            return
        if self.proto.reroute_on_retry:
            kind, final = f.kind, f.final
            if kind == COLLECT and not (self.role[i] == MEMBER and self.ch[i] != i
                                        and self.alive[self.ch[i]]):
                kind, final = UPSTREAM, self.sink
            elif kind == COLLECT:
                final = self.ch[i]
            t0 = time.perf_counter()
            nh = self.proto.next_hop(i, kind, final, len(f.readings))
            self.t_dec += time.perf_counter() - t0
            self.n_dec += 1
            if nh is None or nh < 0 or not self.alive[nh]:
                f.kind, f.final = kind, final
                self._requeue(i, f)
                return
            f.kind, f.final, f.dst = kind, final, nh
        f.nb = 0
        f.be = R.MAC_MIN_BE
        f.corrupt = False
        self.mac[i] = CSMA
        self.tok[i] += 1
        extra = self.proto.retry_delay(i, f)      # 0 for every baseline
        self.schedule(self.now + extra + self.rng.randrange(1 << f.be) * R.UNIT_BACKOFF,
                      EV_BACKOFF, i, self.tok[i])

    # --------------------------------------------------------------- network
    def _receive(self, r: int, f: Frame) -> None:
        if r == self.sink:
            self._deliver(f)
            return
        k = len(f.readings)
        self.cnt_in[r] += k
        # copies are needed: the sender keeps its frame until the ACK arrives
        rds = [[rd[0], rd[1], rd[2], rd[3], rd[4] + 1, 0.0] for rd in f.readings]
        if f.kind == COLLECT and f.final == r:
            self.cnt_in_collect[r] += k
            self._accept(r, rds, own=False)
            return
        ttl = [rd for rd in rds if rd[HOPS] > TTL]
        if ttl:
            self._drop(ttl, "ttl")
            rds = [rd for rd in rds if rd[HOPS] <= TTL]
        if rds:
            self._accept(r, rds, own=False)

    def _deliver(self, f: Frame) -> None:
        now = self.now
        ids = self.delivered_ids
        w0, w1 = self.warm, self.t_stop
        for rd in f.readings:
            rid = rd[RID]
            if rid in ids:
                self.dup_at_sink += 1
                continue
            ids.add(rid)
            self.del_total += 1
            self.del_sec += 1
            tg = rd[TGEN]
            if w0 <= tg < w1:
                c = rd[CLS]
                self.del_win[c] += 1
                self.del_src[rd[ORIG]] += 1
                d = now - tg
                self.delays.append(d)
                if c == 1:
                    self.delays_c.append(d)
                self.hops_del.append(rd[HOPS] + 1)

    def _generate(self, i: int, cls: int) -> None:
        now = self.now
        rd = [next(self._rid), i, now, cls, 0, now]
        inw = self.warm <= now < self.t_stop
        self.cnt_gen[i] += 1
        if inw:
            self.gen_win[cls] += 1
            self.gen_src[i] += 1
        t0 = time.perf_counter()
        adm = self.proto.admit(i, cls)
        self.t_dec += time.perf_counter() - t0
        self.n_dec += 1
        if not adm:
            if inw:
                self.sup_win[cls] += 1
            return
        self.cnt_adm[i] += 1
        if inw:
            self.adm_win[cls] += 1
        self._accept(i, [rd], own=True)

    # ------------------------------------------------------------------ tick
    def _tick(self) -> None:
        en = self.energy
        bits = R.BEACON_BITS
        e_btx = en.tx(bits, self.top.R)
        e_brx = en.rx(bits)
        idle = en.idle_power
        al = self.alive
        for i in range(self.n):
            if al[i]:
                k = 0
                for j in self.nbr[i]:
                    if al[j]:
                        k += 1
                self.charge(i, e_btx + k * e_brx + idle)
                self.ctrl_bytes += bits // 8
        for i in range(self.n):
            if al[i]:
                self.adv_q[i] = self.qlen[i]
        self.proto.tick(self.now)
        self.ts_t.append(self.now)
        self.ts_alive.append(self.alive_count)
        self.ts_energy.append(sum(self.E[i] for i in range(self.n) if al[i]))
        self.ts_deliv.append(self.del_sec)
        self.del_sec = 0
        q = [self.qlen[i] for i in range(self.n) if al[i]]
        self.ts_q.append(sum(q) / len(q) if q else 0.0)
        self.ts_heads.append(sum(1 for i in range(self.n) if al[i] and self.role[i] == HEAD))
        if self.now + 1.0 <= self.T + 1e-9:
            self.schedule(self.now + 1.0, EV_TICK, -1)

    # ------------------------------------------------------------------- run
    def _schedule_traffic(self) -> None:
        lam = self.lam
        for i in range(self.n):
            if lam > 0:
                self.schedule(self.rng_node[i].expovariate(lam), EV_GEN, i)
        sc = self.scen
        if sc.get("events"):
            erng = random.Random(self.seed * 13 + 5)
            t = erng.expovariate(sc["ev_rate"])
            while t < self.T:
                cx, cy = erng.uniform(0, sc["area"]), erng.uniform(0, sc["area"])
                rad2 = sc["ev_radius"] ** 2
                t_end = min(t + sc["ev_dur"], self.T)
                for i in range(self.n):
                    x, y = self.pos[i]
                    if (x - cx) ** 2 + (y - cy) ** 2 <= rad2:
                        tc = t + erng.expovariate(sc["ev_lam"])
                        while tc < t_end:
                            self.schedule(tc, EV_GENC, i)
                            tc += erng.expovariate(sc["ev_lam"])
                t += erng.expovariate(sc["ev_rate"])

    def run(self) -> dict:
        wall0 = time.perf_counter()
        self.refresh_routes()
        self.proto.setup(self)
        self._schedule_traffic()
        self.schedule(1.0, EV_TICK, -1)
        heap = self.heap
        pop = heapq.heappop
        push = heapq.heappush
        T = self.T
        rng_node = self.rng_node
        lam = self.lam
        seq = self._seq
        while heap:
            t, _, et, i, a = pop(heap)
            if t > T:
                break
            self.now = t
            if et == EV_BACKOFF:
                self._backoff_end(i, a)
            elif et == EV_TXSTART:
                self._tx_start(i, a)
            elif et == EV_TXEND:
                self._tx_end(i, a)
            elif et == EV_ACK:
                self._ack(i, a[0], a[1])
            elif et == EV_GEN:
                if self.alive[i]:
                    self._generate(i, 0)
                    push(heap, (t + rng_node[i].expovariate(lam), next(seq), EV_GEN, i, None))
            elif et == EV_TRY:
                if a == self.tok[i] and self.alive[i] and self.mac[i] == HOLD:
                    self._try_send(i)
            elif et == EV_GENC:
                if self.alive[i]:
                    self._generate(i, 1)
            elif et == EV_TICK:
                self._tick()
        self.now = T
        res = self.results()
        res["wall_s"] = time.perf_counter() - wall0
        return res

    # --------------------------------------------------------------- results
    def results(self) -> dict:
        n = self.n
        win = self.t_stop - self.warm
        gen = self.gen_win[0] + self.gen_win[1]
        dl = self.del_win[0] + self.del_win[1]
        adm = self.adm_win[0] + self.adm_win[1]
        nan = float("nan")

        def pct(a, q):
            return float(np.percentile(a, q)) if len(a) else nan

        d = np.asarray(self.delays)
        dc = np.asarray(self.delays_c)
        ratios = [self.del_src[i] / self.gen_src[i] for i in range(n) if self.gen_src[i] > 0]
        if ratios:
            s1 = sum(ratios)
            s2 = sum(x * x for x in ratios)
            jain = (s1 * s1) / (len(ratios) * s2) if s2 > 0 else 0.0
        else:
            jain = nan
        cfs = []
        for key, c in self.link_ok.items():
            i, j = divmod(key, self.N)
            cap = self.lq[i][j] / self.t_cycle
            cfs.append((c / win) / cap)
        cfs = np.asarray(cfs) if cfs else np.asarray([0.0])
        # loss attribution: last recorded reason of every window reading never delivered
        drops = {k: 0 for k in DROP_REASONS}
        ids = self.delivered_ids
        for rid, why in self.drop_reason.items():
            if rid not in ids:
                drops[why] += 1
        used = sum(self.used[:n])
        alive_E = [self.E[i] for i in range(n) if self.alive[i]]
        res = {
            "gen": gen, "gen_crit": self.gen_win[1], "admitted": adm,
            "suppressed": self.sup_win[0] + self.sup_win[1],
            "delivered": dl, "delivered_crit": self.del_win[1],
            "gdr": dl / gen if gen else nan,
            "gdr_crit": self.del_win[1] / self.gen_win[1] if self.gen_win[1] else nan,
            "pdr_admitted": dl / adm if adm else nan,
            "tdr_05": float((d <= 0.5).sum()) / gen if gen else nan,
            "tdr_1": float((d <= 1.0).sum()) / gen if gen else nan,
            "tdr_2": float((d <= 2.0).sum()) / gen if gen else nan,
            "throughput_rps": dl / win,
            "goodput_kbps": dl * R.READING_BYTES * 8 / win / 1000.0,
            "delay_mean": float(d.mean()) if len(d) else nan,
            "delay_median": pct(d, 50), "delay_p95": pct(d, 95), "delay_p99": pct(d, 99),
            "delay_crit_mean": float(dc.mean()) if len(dc) else nan,
            "delay_crit_p95": pct(dc, 95),
            "hops_mean": float(np.mean(self.hops_del)) if self.hops_del else nan,
            "jain": jain,
            "energy_used_J": used,
            "energy_per_reading_mJ": 1000.0 * used / max(self.del_total, 1),
            "energy_remaining_mean": float(np.mean(alive_E)) if alive_E else 0.0,
            "energy_remaining_min": float(np.min(alive_E)) if alive_E else 0.0,
            "fnd": self.fnd if self.fnd is not None else nan,
            "hnd": self.hnd if self.hnd is not None else nan,
            "alive_frac": self.alive_count / n,
            "cf_p50": float(np.percentile(cfs, 50)), "cf_p90": float(np.percentile(cfs, 90)),
            "cf_p99": float(np.percentile(cfs, 99)), "cf_max": float(np.max(cfs)),
            "cf_frac_gt07": float(np.mean(cfs > 0.7)), "n_links": int(len(self.link_ok)),
            "tx_frames": self.tx_frames,
            "frames_per_reading": self.tx_frames / max(self.del_total, 1),
            "readings_per_frame": self.tx_readings / max(self.tx_frames, 1),
            "dup_at_sink": self.dup_at_sink,
            "ctrl_kB": self.ctrl_bytes / 1000.0, "data_kB": self.data_bytes / 1000.0,
            "dec_us": 1e6 * self.t_dec / max(self.n_dec, 1),
            "dec_ms_per_s": 1e3 * self.t_dec / self.T,
            "sink_ms_per_round": 1e3 * self.t_sink / max(self.n_rounds, 1),
            "sink_ms_per_s": 1e3 * self.t_sink / self.T,
            "n_rounds": self.n_rounds,
            "heads_mean": float(np.mean(self.ts_heads)) if self.ts_heads else nan,
            "queue_mean": float(np.mean(self.ts_q)) if self.ts_q else nan,
            "drop_buf_head": self.drop_buf_head,
            "queue_max": self.qmax,
            "sink_coll_frac": self.cnt_rx_coll[self.sink] / max(self.cnt_rx_att[self.sink], 1),
            "sink_util": self.busy_acc[self.sink] / self.T,
            "coll_frac_all": sum(self.cnt_rx_coll) / max(sum(self.cnt_rx_att), 1),
        }
        for k in DROP_REASONS:
            res[f"drop_{k}"] = drops[k]
        res["in_flight"] = gen - dl - res["suppressed"] - sum(drops.values())
        return res

    def timeseries(self) -> dict:
        return {"t": self.ts_t, "alive": self.ts_alive, "energy": self.ts_energy,
                "delivered": self.ts_deliv, "queue": self.ts_q, "heads": self.ts_heads}
