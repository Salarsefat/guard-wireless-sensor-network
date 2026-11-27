"""Mechanism analysis (instrumentation only; no design decision depends on it).

Re-runs GUARD, TwinLoop v1, C-ETX and BCP with their tuned configurations on
TEST2 seeds 101-103 at 1 and 3 readings/s per node, recording for every frame
attempt the sender-receiver distance, whether the receiver is the sink, and
whether the frame was corrupted by interference.  The simulator is wrapped,
not modified: Sim._tx_end is decorated to log attempts before the original
method runs.  Output: results/test2/mechanism.json and
output/figures/fig_mechanism.{pdf,png} and fig_tdr_curve.{pdf,png}.
"""
from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "analysis"))

METHODS = ["guard", "twinloop", "cetx", "bcp", "ganaco"]
SEEDS = [101, 102, 103]
LOADS = [1.0, 3.0]


def run(args):
    method, lam, seed = args
    from wsncc.protocols import make_protocol
    from wsncc.runner import scenario
    from wsncc.sim import Sim
    tuned = json.loads((ROOT / "results" / "tuned_params.json").read_text())
    proto = make_protocol(method, dict(tuned[method]["params"]))
    sim = Sim(scenario(lam=lam, T=300.0), proto, seed)
    log = []
    orig = sim._tx_end

    def wrapped(i, f):
        if sim.warm <= sim.now < sim.t_stop:
            log.append((float(sim.dist[i][f.dst]), int(f.dst == sim.sink), int(bool(f.corrupt)),
                        len(f.readings)))
        return orig(i, f)

    sim._tx_end = wrapped
    res = sim.run()
    a = np.array(log) if log else np.zeros((0, 4))
    sink = a[a[:, 1] == 1]
    other = a[a[:, 1] == 0]
    bins = [0, 40, 57.7, 70, 80, 90, 100, 200]
    hist = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = sink[(sink[:, 0] >= lo) & (sink[:, 0] < hi)]
        hist.append({"lo": lo, "hi": hi, "n": int(len(sel)),
                     "coll": float(sel[:, 2].mean()) if len(sel) else None})
    taus = np.logspace(-2.3, 0.6, 60)
    dl = np.sort(np.asarray(sim.delays))
    gen = max(sim.gen_win[0] + sim.gen_win[1], 1)
    tdr_curve = (np.searchsorted(dl, taus, side="right") / gen).tolist()
    return {"method": method, "lam": lam, "seed": seed, "taus": taus.tolist(), "tdr_curve": tdr_curve,
            "sink_attempts": int(len(sink)), "sink_mean_len": float(sink[:, 0].mean()) if len(sink) else None,
            "sink_frac_free": float((sink[:, 0] <= 150 / 2.6).mean()) if len(sink) else None,
            "sink_coll": float(sink[:, 2].mean()) if len(sink) else None,
            "other_coll": float(other[:, 2].mean()) if len(other) else None,
            "sink_readings_per_frame": float(sink[:, 3].mean()) if len(sink) else None,
            "sink_hist": hist, "tdr_1": res["tdr_1"], "drop_retry": res["drop_retry"],
            "sink_lengths": sink[:, 0].round(1).tolist()}


def main():
    cache = ROOT / "results" / "test2" / "mechanism.json"
    if cache.exists() and "--rerun" not in sys.argv:
        out = json.loads(cache.read_text())
    else:
        jobs = [(m, lam, s) for m in METHODS for lam in LOADS for s in SEEDS]
        with Pool(2) as pool:
            out = pool.map(run, jobs)
        cache.write_text(json.dumps(out))
    summ = {}
    for m in METHODS:
        for lam in LOADS:
            rs = [r for r in out if r["method"] == m and r["lam"] == lam]
            summ[f"{m}@{lam:g}"] = {k: float(np.mean([r[k] for r in rs])) for k in
                                    ("sink_mean_len", "sink_frac_free", "sink_coll", "other_coll",
                                     "sink_readings_per_frame", "tdr_1", "drop_retry")}
            print(m, lam, {k: round(v, 3) for k, v in summ[f"{m}@{lam:g}"].items()})
    (ROOT / "results" / "test2" / "mechanism_summary.json").write_text(json.dumps(summ, indent=1))
    # figure: distribution of last-hop lengths (attempts into the sink) and collision rate by length
    import matplotlib.pyplot as plt
    from common import COLOR, FIG, LABEL, MARKER, TEXT2, setup_mpl
    setup_mpl()
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.6))
    bins = np.arange(0, 121, 5)
    for m in METHODS:
        L = np.sort(np.concatenate([r["sink_lengths"] for r in out if r["method"] == m and r["lam"] == 3.0]))
        axs[0].step(L, np.arange(1, len(L) + 1) / len(L), where="post", color=COLOR[m], label=LABEL[m],
                    lw=2.0 if m == "guard" else 1.2)
        xs, ys = [], []
        for k, b in enumerate(out[0]["sink_hist"]):
            n = sum(r["sink_hist"][k]["n"] for r in out if r["method"] == m and r["lam"] == 3.0)
            c = sum((r["sink_hist"][k]["coll"] or 0) * r["sink_hist"][k]["n"] for r in out
                    if r["method"] == m and r["lam"] == 3.0)
            if n >= 30:
                xs.append((b["lo"] + min(b["hi"], 120)) / 2)
                ys.append(c / n)
        axs[1].plot(xs, ys, marker=MARKER[m], color=COLOR[m], label=LABEL[m], lw=2.0 if m == "guard" else 1.2)
    for ax in axs:
        ax.axvline(150 / 2.6, color=TEXT2, ls=":", lw=0.8)
        ax.set_xlabel("length of the link into the sink [m]")
    axs[0].set_ylabel("cumulative fraction of attempts")
    axs[1].set_ylabel("fraction corrupted by interference")
    axs[0].set_title("(a) length of attempts into the sink, 3 readings/s", loc="left")
    axs[1].set_title("(b) collision rate by length, 3 readings/s", loc="left")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "fig_mechanism.pdf")
    fig.savefig(FIG / "fig_mechanism.png", dpi=600)
    plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for ax, lam in zip(axs, LOADS):
        for m in METHODS:
            rs = [r for r in out if r["method"] == m and r["lam"] == lam]
            curve = np.mean([r["tdr_curve"] for r in rs], axis=0)
            ax.plot(rs[0]["taus"], curve, color=COLOR[m], label=LABEL[m], lw=2.0 if m == "guard" else 1.2)
        ax.set_xscale("log")
        ax.set_xlabel("deadline $\\tau$ [s]")
        ax.set_ylim(0, 1.02)
        ax.set_title(f"({'a' if lam == LOADS[0] else 'b'}) {lam:g} reading{'s' if lam != 1 else ''}/s per node", loc="left")
    axs[0].set_ylabel("fraction delivered within $\\tau$")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(FIG / "fig_tdr_curve.pdf")
    fig.savefig(FIG / "fig_tdr_curve.png", dpi=600)
    # crossing points: smallest tau at which GUARD's curve is at least every baseline's
    cross = {}
    for lam in LOADS:
        g = np.mean([r["tdr_curve"] for r in out if r["method"] == "guard" and r["lam"] == lam], axis=0)
        taus = out[0]["taus"]
        for m in METHODS[1:]:
            b = np.mean([r["tdr_curve"] for r in out if r["method"] == m and r["lam"] == lam], axis=0)
            idx = [k for k in range(len(taus)) if all(g[j] >= b[j] for j in range(k, len(taus)))]
            cross[f"{m}@{lam:g}"] = float(taus[idx[0]]) if idx else None
    summ["crossing_tau"] = cross
    (ROOT / "results" / "test2" / "mechanism_summary.json").write_text(json.dumps(summ, indent=1))
    print("crossing:", cross)


if __name__ == "__main__":
    main()
