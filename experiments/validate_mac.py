"""Sanity checks of the packet-level simulator on hand-built topologies.

1. Saturated single link: measured service rate vs the analytical idle-channel
   service time of a full frame (mean initial backoff + CCA + turnaround +
   airtime + turnaround + ACK).
2. Two senders and one receiver, with the senders either inside each other's
   carrier-sense range or hidden from each other: fraction of frames corrupted
   by interference at the receiver as the load grows.

These runs use no TEST/VAL seed sets (seeds 1..5 of a separate stream on
synthetic topologies) and do not involve any protocol under comparison.

Usage: python experiments/validate_mac.py
Writes results/validation/mac_validation.json and output/tables/tab_mac_validation.tex
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wsncc import radio as R  # noqa: E402
from wsncc.protocols.base import Protocol  # noqa: E402
from wsncc.runner import scenario  # noqa: E402
from wsncc.sim import Sim  # noqa: E402
from wsncc.topology import Topology  # noqa: E402


class Direct(Protocol):
    """Flat minimum-ETX forwarding without any congestion control."""
    name = "direct"
    clustered = False

    def next_hop(self, i, kind, final, k=1):
        p = self.sim.parent[i]
        return p if p >= 0 else None


def topo(pos, links, hear, prr=0.999):
    """pos: list of (x, y) with the sink last; links: routing pairs; hear: carrier-sense pairs."""
    N = len(pos)
    nbr = [[] for _ in range(N)]
    pr = [dict() for _ in range(N)]
    dist = [dict() for _ in range(N)]
    intf = [[] for _ in range(N)]
    for a, b in links:
        d = float(np.hypot(pos[a][0] - pos[b][0], pos[a][1] - pos[b][1]))
        for u, v in ((a, b), (b, a)):
            nbr[u].append(v)
            pr[u][v] = prr
            dist[u][v] = d
    for a, b in set(map(tuple, map(sorted, list(hear) + list(links)))):
        intf[a].append(b)
        intf[b].append(a)
    lq = [{v: pr[u][v] * pr[v][u] for v in nbr[u]} for u in range(N)]   # bidirectional ETX, as in build_topology
    return Topology(n=N - 1, pos=pos, R=100.0, nbr=nbr, prr=pr, lq=lq, dist=dist, intf=intf)


def run(top, lam, seed, T=60.0):
    sc = scenario(n=top.n, lam=lam, T=T, warm=5.0, drain=2.0)
    sim = Sim(sc, Direct(), seed, topology=top)
    res = sim.run()
    res["sink_att"] = sim.cnt_rx_att[sim.sink]
    res["sink_coll"] = sim.cnt_rx_coll[sim.sink]
    return res


def main():
    out = {}
    # 1. saturated single link (50 m)
    t1 = topo([(50.0, 0.0), (0.0, 0.0)], links=[(0, 1)], hear=[])
    bits = R.frame_bits(4)
    t_cycle = R.service_cycle(bits)
    analytic = 4.0 / t_cycle
    meas = []
    for s in range(1, 6):
        r = run(t1, lam=2000.0, seed=10_000 + s)
        meas.append(r["throughput_rps"])
    out["single_link"] = {"frame_bytes": bits // 8, "t_cycle_ms": 1e3 * t_cycle,
                          "analytic_rps": analytic, "measured_rps_mean": float(np.mean(meas)),
                          "measured_rps_sd": float(np.std(meas, ddof=1)),
                          "ratio": float(np.mean(meas) / analytic)}
    # 2. two senders 90 m from the receiver, 180 m apart (hidden) or 90 m apart (in carrier-sense range)
    pos_h = [(-90.0, 0.0), (90.0, 0.0), (0.0, 0.0)]
    pos_v = [(-45.0, 77.94), (45.0, 77.94), (0.0, 0.0)]
    hidden = topo(pos_h, links=[(0, 2), (1, 2)], hear=[])
    visible = topo(pos_v, links=[(0, 2), (1, 2)], hear=[(0, 1)])
    rows = []
    for lam in (5.0, 20.0, 50.0, 100.0):
        for name, tp in (("hidden", hidden), ("visible", visible)):
            cf, gdr = [], []
            for s in range(1, 6):
                r = run(tp, lam=lam, seed=20_000 + s)
                cf.append(r["sink_coll"] / max(r["sink_att"], 1))
                gdr.append(r["gdr"])
            rows.append({"lam": lam, "case": name, "coll_frac": float(np.mean(cf)),
                         "gdr": float(np.mean(gdr))})
    out["two_senders"] = rows
    vd = ROOT / "results" / "validation"
    vd.mkdir(parents=True, exist_ok=True)
    (vd / "mac_validation.json").write_text(json.dumps(out, indent=1))
    sl = out["single_link"]
    body = "\n".join(
        f"{r['lam']:g} & {100 * r['coll_frac']:.1f} & {r['gdr']:.3f} & "
        f"{100 * next(x for x in rows if x['lam'] == r['lam'] and x['case'] == 'visible')['coll_frac']:.1f} & "
        f"{next(x for x in rows if x['lam'] == r['lam'] and x['case'] == 'visible')['gdr']:.3f} \\\\"
        for r in rows if r["case"] == "hidden")
    tex = r"""\begin{table}[H]
\caption{Simulator sanity checks on hand-built topologies (5 seeds each, 60\,s). Top: saturated single 50\,m link with full frames of """ + str(sl["frame_bytes"]) + r"""\,bytes; the analytical rate is $K/t_\mathrm{cyc}$ with $t_\mathrm{cyc}=""" + f"{sl['t_cycle_ms']:.3f}" + r"""$\,ms. Bottom: two senders at 90\,m from a common receiver, hidden from each other (180\,m apart, beyond the 150\,m carrier-sense range) or within carrier-sense range (90\,m apart); per-sender load in readings/s.\label{tab:macval}}
\small
\begin{tabularx}{\textwidth}{lCCCC}
\toprule
\multicolumn{5}{l}{\textbf{Saturated single link}}\\
 & \multicolumn{2}{c}{Analytical (readings/s)} & \multicolumn{2}{c}{Simulated (readings/s)}\\
 & \multicolumn{2}{c}{""" + f"{sl['analytic_rps']:.1f}" + r"""} & \multicolumn{2}{c}{""" + f"{sl['measured_rps_mean']:.1f}\\,$\\pm$\\,{sl['measured_rps_sd']:.1f}" + r"""}\\
\midrule
\multicolumn{5}{l}{\textbf{Two senders, one receiver}}\\
 & \multicolumn{2}{c}{Hidden senders} & \multicolumn{2}{c}{Senders in carrier-sense range}\\
\textbf{Load} & \textbf{Collided (\%)} & \textbf{GDR} & \textbf{Collided (\%)} & \textbf{GDR}\\
\midrule
""" + body + r"""
\bottomrule
\end{tabularx}
\end{table}
"""
    tdir = ROOT / "output" / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    (tdir / "tab_mac_validation.tex").write_text(tex)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
