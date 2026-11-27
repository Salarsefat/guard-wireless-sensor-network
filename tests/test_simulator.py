"""Invariant and sanity tests for the simulator and protocols (pytest)."""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wsncc import radio as R  # noqa: E402
from wsncc.clustering import capacitated_milp, greedy_capacitated  # noqa: E402
from wsncc.protocols import REGISTRY, make_protocol  # noqa: E402
from wsncc.runner import run_one, scenario  # noqa: E402
from wsncc.sim import Sim  # noqa: E402
from wsncc.topology import build_topology, etx_distances  # noqa: E402

SHORT = {"T": 60.0, "warm": 10.0, "drain": 5.0, "lam": 0.5, "n": 40, "area": 320.0}


def test_frame_sizes_respect_802154():
    assert R.frame_bits(4, 2) // 8 - R.PHY_OVERHEAD <= R.MAX_PSDU
    assert R.max_pack(2) == 5
    with pytest.raises(ValueError):
        R.frame_bits(6, 2)


def test_timing_constants():
    assert math.isclose(R.UNIT_BACKOFF, 320e-6)
    assert math.isclose(R.T_ACKWAIT, 864e-6)
    assert math.isclose(R.airtime(250_000), 1.0)


def test_first_order_energy_crossover():
    e = R.FirstOrderEnergy()
    assert math.isclose(e.d0, math.sqrt(10e-12 / 0.0013e-12))
    assert e.tx(1000, 50.0) == pytest.approx(1000 * (50e-9 + 10e-12 * 2500))
    assert e.rx(1000) == pytest.approx(50e-6)


def test_topology_connected_and_deterministic():
    a = build_topology(60, 400.0, 100.0, seed=7)
    b = build_topology(60, 400.0, 100.0, seed=7)
    assert a.pos == b.pos
    H, parent = etx_distances(a, [True] * 61)
    assert all(h < math.inf for h in H)
    assert parent[a.sink] == -1


@pytest.mark.parametrize("method", sorted(REGISTRY))
def test_reading_conservation(method):
    """Every generated window reading is delivered, suppressed, dropped or in flight."""
    params = {"qtable": [[0.0, 0.0, 0.0]] * 48} if method == "rlcc" else {}
    r = run_one(method, params, SHORT, seed=11)
    assert r["gen"] > 0
    accounted = r["delivered"] + r["suppressed"] + sum(
        r[f"drop_{k}"] for k in ("buf", "caf", "retry", "ttl", "dead"))
    assert r["in_flight"] == r["gen"] - accounted
    assert r["in_flight"] >= 0
    assert 0.0 <= r["gdr"] <= 1.0
    assert r["tdr_1"] <= r["gdr"] + 1e-12


def test_common_random_numbers_across_methods():
    a = run_one("cetx", {}, SHORT, seed=3)
    b = run_one("twinloop", {}, SHORT, seed=3)
    assert a["gen"] == b["gen"]          # identical traffic realisations


def test_single_link_delivery_is_near_prr():
    """Two nodes, no contention: almost every reading reaches the sink."""
    sc = scenario(n=1, area=60.0, T=120.0, warm=5.0, drain=5.0, lam=2.0)
    r = Sim(sc, make_protocol("cetx", {}), seed=5).run()
    assert r["gdr"] > 0.97


def test_twinloop_queue_bound_holds_without_overflow():
    r = run_one("twinloop", {"Va": 8}, {**SHORT, "lam": 2.0}, seed=4)
    assert r["drop_buf"] == 0


def test_milp_matches_or_beats_greedy():
    nodes = list(range(6))
    f = {j: 3.0 for j in nodes}
    pairs = [(i, i, 0.0) for i in nodes] + [(i, j, 1.0 + abs(i - j)) for i in nodes for j in nodes
                                            if i != j and abs(i - j) <= 2]
    lam = {i: 1.0 for i in nodes}
    cap = {j: 3.0 for j in nodes}
    h1, c1, info = capacitated_milp(nodes, f, pairs, lam, cap)
    h2, c2, info2 = greedy_capacitated(nodes, f, pairs, lam, cap)
    assert info["status"] == 0
    assert info["obj"] <= info2["obj"] + 1e-9
    for j in h1:  # capacity respected
        assert sum(lam[i] for i, h in c1.items() if h == j) + lam[j] <= cap[j] + 1e-6
