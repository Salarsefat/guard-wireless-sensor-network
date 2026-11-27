"""Tables, figures and the number summary of version 2 (GUARD), TEST2 seeds.

Usage:  python analysis/v2.py [tables] [figures]      (default: both)
Reads results/test2/E*.jsonl and results/tuned_params.json; writes
output/tables/*.tex, output/figures/*.{pdf,png}, results/test2/summary_v2.json and
results/test2/paired_tests_v2_*.csv.  Every number quoted in the manuscript is
taken from summary_v2.json or from the tables generated here.
"""
from __future__ import annotations

import json
import math
import sys

import numpy as np
import pandas as pd

from common import (BASELINES, COLOR, FIG, LABEL, MARKER, METHODS, PLOT, RES, ROOT, TAB, TEXT, TEXT2, GRID,
                    friedman, mean_ci, paired_tests, setup_mpl)

SUMMARY: dict = {}
TAB.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

# metric -> (label, higher_is_better, digits)
METRICS = {
    "tdr_05": ("TDR(0.5 s)", True, 3), "tdr_1": ("TDR(1 s)", True, 3), "tdr_2": ("TDR(2 s)", True, 3),
    "gdr": ("delivery ratio", True, 3), "pdr_admitted": ("delivery ratio of admitted readings", True, 3),
    "throughput_rps": ("throughput [readings/s]", True, 1),
    "delay_mean": ("mean delay [s]", False, 3), "delay_median": ("median delay [s]", False, 3),
    "delay_p95": ("95th-percentile delay [s]", False, 3), "delay_p99": ("99th-percentile delay [s]", False, 3),
    "energy_per_reading_mJ": ("energy per reading [mJ]", False, 3), "jain": ("Jain index", True, 3),
    "lost_frac": ("loss ratio", False, 3),
    "coll_frac_all": ("collided frames (all receivers)", False, 3),
    "sink_coll_frac": ("collided frames (sink)", False, 3),
    "frames_per_reading": ("frames per delivered reading", False, 2),
    "ctrl_per_s": ("control bytes per node and s", False, 2),
    "ovh_per_s": ("protocol overhead bytes per node and s", False, 2),
    "drop_retry": ("readings lost to retry exhaustion per run", False, 0),
    "drop_buf": ("readings lost to buffer overflow per run", False, 0),
}
HDR_EXTRA = {"guard": 0, "twinloop": 2, "bcp": 2, "cbpr": 2}
# metrics of the win/tie/loss matrix (one row each)
WTL = ["tdr_05", "tdr_1", "tdr_2", "gdr", "throughput_rps", "delay_mean", "delay_p95", "delay_p99",
       "energy_per_reading_mJ", "jain", "lost_frac", "coll_frac_all", "sink_coll_frac",
       "frames_per_reading", "ctrl_per_s", "ovh_per_s"]


def wide(tex: str) -> str:
    tex = tex.replace("\\begin{table}[H]\n", "\\begin{table}[H]\n\\begin{adjustwidth}{-\\extralength}{0cm}\n", 1)
    tex = tex.replace("\\begin{tabularx}{\\textwidth}", "\\begin{tabularx}{\\fulllength}", 1)
    tex = tex.replace("\\end{table}", "\\end{adjustwidth}\n\\end{table}", 1)
    return tex


# Tables of the article may float to the top or bottom of a page ([!htb]), so that
# tables printed in the article float ([!htb]); the other generated tables stay in place ([H]).
ARTICLE_TABLES = {"tab_guard_params", "tab_e1tdr", "tab_e1wtl", "tab_complexity", "tab_e3tdr", "tab_e4", "tab_e5"}


def write_table(path, tex: str) -> None:
    if path.stem in ARTICLE_TABLES:
        tex = tex.replace("\\begin{table}[H]", "\\begin{table}[!htb]", 1)
    path.write_text(tex)


EXPECTED = {"E1": 810, "E2": 360, "E3": 540, "E4": 90, "E5": 360, "E6": 290, "E7": 140, "E8": 270}


def load(exp: str) -> pd.DataFrame | None:
    path = RES / f"{exp}.jsonl"
    if not path.exists():
        return None
    recs = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    if len(recs) < EXPECTED.get(exp, 0):
        print(f"[skip] {exp}: {len(recs)}/{EXPECTED[exp]} runs (incomplete)", file=sys.stderr)
        return None
    df = pd.DataFrame(recs)
    bad = df[df.get("status", "ok") != "ok"]
    if len(bad):
        print(f"[warn] {exp}: {len(bad)} failed runs", file=sys.stderr)
    df = df[df["status"] == "ok"].copy()
    df["lost_frac"] = 1.0 - df["gdr"]
    win = (df["T"] - 40.0)
    df["ctrl_per_s"] = 1000.0 * df["ctrl_kB"] / (df["T"] * df["n"])
    # protocol-specific overhead: control bytes plus extra data-frame header bytes
    df["ovh_per_s"] = (1000.0 * df["ctrl_kB"] + df["method"].map(HDR_EXTRA).fillna(0) * df["tx_frames"]) / (df["T"] * df["n"])
    df["timely"] = df["tdr_1"] * df["gen"]
    SUMMARY.setdefault("runs", {})[exp] = int(len(df))
    return df


def fmt(m, h, d=3):
    if m != m:
        return "--"
    if h != h:
        return f"{m:.{d}f}"
    return f"{m:.{d}f}\\,$\\pm$\\,{h:.{d}f}"


def sig_symbol(r, higher_better):
    if r is None or r["p_holm"] >= 0.05:
        return "$^{\\circ}$"
    better = (r["median_diff"] > 0) == higher_better
    return "$^{\\blacktriangle}$" if better else "$^{\\triangledown}$"


def all_tests(df, group, metrics, ref="guard"):
    out = []
    for m in metrics:
        t = paired_tests(df, m, group, ref=ref, methods=METHODS)
        if len(t):
            out.append(t)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def verdict(row, higher_better):
    if row["p_holm"] < 0.05:
        return "win" if (row["median_diff"] > 0) == higher_better else "loss"
    return "tie"


# --------------------------------------------------------------------- tables
PARAM_TEX = {"kappa": r"$\kappa$", "u0": "$u_0$", "beta": r"$\beta$", "cE": "$c_E$", "cR": "$c_R$",
             "ebal": r"$\omega$", "nlr": r"$N_\mathrm{r}$", "rdel": "$b$", "hyst": r"$\zeta$",
             "alpha_s": r"$\alpha$", "h": "$h$",
             "Va": "$V_a$", "Vf": "$V_f$", "a_p": "$a_p$", "eta": r"$\eta$", "f0": "$f_0$",
             "lifo": "LIFO", "phi_target": r"$\phi^*$", "tau_max": r"$\tau_{\max}$",
             "theta": r"$\theta$", "theta0": r"$\theta_0$", "Tc": "$T_c$", "k": "$k$", "V": "$V$",
             "hold": "hold", "Q": "$Q$", "rho": r"$\rho$", "cf_thr": "CF thr.", "Tctrl": r"$T_\mathrm{ctrl}$",
             "q_low": r"$q_\mathrm{low}$", "w_l": "$w_l$", "w_b": "$w_b$", "c_buf": r"$c_\mathrm{buf}$",
             "c_cca": r"$c_\mathrm{cca}$", "c_cf": r"$c_\mathrm{cf}$", "gain": "$g$", "alpha": r"$\alpha$",
             "gamma": r"$\gamma$"}


def _param_tex(k, v):
    name = PARAM_TEX.get(k, k.replace("_", r"\_"))
    if isinstance(v, bool):
        return f"{name}: {'yes' if v else 'no'}"
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    return f"{name}\\,=\\,{v}"


def table_tuned():
    t = json.loads((ROOT / "results" / "tuned_params.json").read_text())
    rows = []
    for m in METHODS:
        b = t[m]
        sel = ", ".join(_param_tex(k, b["params"][k]) for k in sorted(b["params"])) or "default setting"
        rows.append(f"{LABEL[m]} & {b['config_index']} & {b['n_tied']} & {b['tdr_1']:.3f} & "
                    f"{b['delay_p95']:.3f} & {{\\raggedright\\footnotesize {sel}}} \\\\")
        SUMMARY.setdefault("tuning", {})[m] = {k: b[k] for k in ("config_index", "tdr_1", "gdr", "delay_p95",
                                                                  "n_tied", "best_tdr_1", "params")}
    tex = (r"""\begin{table}[H]
\caption{Configurations selected by the equal-budget random search (12 configurations per method, validation seeds 2001--2002, loads 0.5/1/2/3 readings/s per node). Objective: mean TDR(1\,s); configurations within 0.5\,pp of the best are tied and the tie is broken by the lowest mean 95th-percentile delay. Configuration 0 is each method's default.\label{tab:tuned}}
\footnotesize
\begin{tabularx}{\textwidth}{lccccX}
\toprule
\textbf{Method} & \textbf{Config.} & \textbf{Tied} & \textbf{Val. TDR(1\,s)} & \textbf{Val. p95 [s]} & \textbf{Selected hyper-parameters (non-default)}\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabularx}
\end{table}
""")
    write_table(TAB / "tab_tuned.tex", wide(tex))


def table_guard_params():
    from wsncc.protocols.guard import GUARD
    t = json.loads((ROOT / "results" / "tuned_params.json").read_text())["guard"]["params"]
    d = GUARD.defaults
    rows = [
        (r"$\alpha$", "alpha_s", "weight of the link estimator (per attempt)"),
        (r"$\zeta$", "hyst", "parent hysteresis"),
        (r"$\omega$", "ebal", "energy-balance weight of the routing metric"),
        ("$h$", "h", "holding cost per reading and second (prices are relative to $h$)"),
        (r"$\kappa$", "kappa", "weight of the congestion price"),
        ("$u_0$", "u0", "busy fraction below which the congestion price is zero"),
        ("$c_E$", "cE", "weight of the energy-scarcity price"),
        ("$c_R$", "cR", "weight of the drain-rate price"),
        (r"$\beta$", "beta", "fraction of the slack that may be spent waiting"),
        (r"$w_{\max}$", "wmax", "absolute cap on waiting [s]"),
        (r"$\delta$", "dhop", "delay budget per expected attempt [s]"),
        (r"$\tau$", "tau", "deadline [s]"),
        (r"$N_\mathrm{r}$", "nlr", "network-layer retransmission rounds"),
        ("$b$", "rdel", "bound of the decorrelation delay [s]"),
        ("$a$", "a", "weight of the per-second averages ($u_i$, $\\Lambda_i$, $P_i$)"),
    ]
    body = []
    for sym, key, desc in rows:
        dv = d[key]
        tv = t.get(key, dv)
        body.append(f"{sym} & {desc} & {dv:g} & {tv:g} \\\\")
    tex = (r"""\begin{table}[H]
\caption{Parameters of GUARD: default (chosen on development seeds) and tuned value (validation seeds; used in all test experiments unless stated otherwise).\label{tab:guardparams}}
\footnotesize
\begin{tabularx}{\textwidth}{lXcc}
\toprule
\textbf{Symbol} & \textbf{Meaning} & \textbf{Default} & \textbf{Tuned}\\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabularx}
\end{table}
""")
    write_table(TAB / "tab_guard_params.tex", tex)


HEAD_NAMES = {"cc2420": "CC2420"}   # column headings as printed in the article


def table_by_load(df, tests, metric, tab_label, caption, group="lam", head_fmt="{:g}", d=3, outdir=None,
                  colsep="2.5pt"):
    label, hb, _ = METRICS[metric]
    groups = sorted(df[group].unique())
    best = {}
    for gv in groups:
        means = {m: df[(df.method == m) & (df[group] == gv)][metric].mean() for m in METHODS}
        best[gv] = (max if hb else min)(means, key=means.get)
    rows = []
    for m in METHODS:
        cells = []
        for gv in groups:
            mm, h, _ = mean_ci(df[(df.method == m) & (df[group] == gv)][metric])
            s_ = fmt(mm, h, d)
            if best[gv] == m:
                s_ = "\\textbf{" + s_.replace("\\,$\\pm$\\,", "}\\,$\\pm$\\,\\textbf{") + "}"
            if m != "guard" and len(tests):
                sub = tests[(tests.metric == metric) & (tests[group] == gv) & (tests.baseline == m)]
                s_ += sig_symbol(sub.iloc[0] if len(sub) else None, hb)
            cells.append(s_)
        rows.append(f"{LABEL[m]} & " + " & ".join(cells) + " \\\\")
    head = " & ".join("\\textbf{" + HEAD_NAMES.get(head_fmt.format(g), head_fmt.format(g)) + "}" for g in groups)
    tex = (r"""\begin{table}[H]
\caption{""" + caption + r"""\label{""" + tab_label + r"""}}
\footnotesize
\setlength{\tabcolsep}{""" + colsep + r"""}
\begin{tabularx}{\textwidth}{l""" + "C" * len(groups) + r"""}
\toprule
\textbf{Method} & """ + head + r""" \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabularx}
\end{table}
""")
    out = outdir or TAB
    name = tab_label.replace("tab:", "tab_").replace("stab:", "stab_").replace(":", "_")
    write_table(out / f"{name}.tex", wide(tex))
    return name


SUPP = TAB / "per_metric"
# one table per metric and experiment (every metric quoted in the article is included)
SUPP_METRICS = {
    "E1": ["tdr_05", "tdr_1", "tdr_2", "gdr", "throughput_rps", "delay_mean", "delay_p95", "delay_p99",
           "energy_per_reading_mJ", "jain", "coll_frac_all", "sink_coll_frac", "frames_per_reading",
           "drop_retry", "drop_buf", "ctrl_per_s", "ovh_per_s"],
    "E2": ["tdr_1", "gdr", "delay_mean", "delay_p95", "energy_per_reading_mJ", "jain", "coll_frac_all",
           "frames_per_reading"],
    "E3": ["tdr_1", "gdr", "delay_mean", "delay_p95", "energy_per_reading_mJ", "jain", "coll_frac_all"],
    "E8": ["tdr_1", "gdr", "delay_p95", "energy_per_reading_mJ"],
}
SUPP_DIGITS = {"throughput_rps": 1, "drop_retry": 0, "drop_buf": 0, "ctrl_per_s": 2, "ovh_per_s": 2,
               "frames_per_reading": 2}
SUPP_SCALE = {"delay_mean": 1000.0, "delay_p95": 1000.0, "delay_p99": 1000.0}


def per_metric_tables():
    """One table per metric and experiment (labels stab:<exp>:<metric>): the full results behind the article."""
    SUPP.mkdir(parents=True, exist_ok=True)
    specs = {"E1": ("lam", "{:g}", "load [readings/s per node]", 15),
             "E2": ("n", "{:d}", "number of sensors", 10),
             "E3": ("scen", "{}", "scenario", 10),
             "E8": ("lam", "{:g}", "load [readings/s per node], default configurations", 10)}
    names = []
    for exp, (group, hf, what, nseeds) in specs.items():
        df = load(exp)
        if df is None:
            continue
        if exp == "E3":
            df["scen"] = _scen(df)
        df = df.copy()
        for k, sc in SUPP_SCALE.items():
            df[k] = df[k] * sc
        tests = all_tests(df, group, SUPP_METRICS[exp])
        parts = []
        for metric in SUPP_METRICS[exp]:
            lab = METRICS[metric][0].replace("[s]", "[ms]") if metric in SUPP_SCALE else METRICS[metric][0]
            cap = (f"{exp}: {lab} by {what}; mean\\,$\\pm$\\,95\\% CI over {nseeds} test seeds; "
                   "best mean in bold; markers: $^{\\blacktriangle}$ GUARD significantly better, $^{\\triangledown}$ significantly worse, $^{\\circ}$ no significant difference (Wilcoxon signed-rank, Holm over 8 baselines).")
            nm = table_by_load(df, tests, metric, f"stab:{exp.lower()}:{metric}", cap, group=group, head_fmt=hf,
                               d=SUPP_DIGITS.get(metric, 1 if metric in SUPP_SCALE else 3), outdir=SUPP,
                               colsep="0.9pt" if metric in SUPP_SCALE else "2.5pt")
            parts.append(f"\\input{{per_metric/{nm}}}")
        (SUPP / f"s_{exp.lower()}_tables.tex").write_text("\n".join(parts) + "\n")
        names.append(exp)
    return names


def table_wtl(tests, group, n_groups, tab_label, caption):
    """Win/tie/loss counts of GUARD vs each baseline, per metric, over the groups."""
    rows = []
    store = {}
    for metric in WTL:
        label, hb, _ = METRICS[metric]
        cells = []
        for b in BASELINES:
            sub = tests[(tests.metric == metric) & (tests.baseline == b)]
            v = [verdict(r, hb) for _, r in sub.iterrows()]
            w, t_, l_ = v.count("win"), v.count("tie"), v.count("loss")
            store.setdefault(metric, {})[b] = {"win": w, "tie": t_, "loss": l_}
            cell = f"{w}/{t_}/{l_}"
            if l_ > 0:
                cell = "\\textit{" + cell + "}"
            cells.append(cell)
        rows.append(f"{label} & " + " & ".join(cells) + " \\\\")
    SUMMARY.setdefault("wtl", {})[tab_label] = store
    head = " & ".join(f"\\textbf{{{LABEL[b]}}}" for b in BASELINES)
    tex = (r"""\begin{table}[H]
\caption{""" + caption + r"""\label{""" + tab_label + r"""}}
\footnotesize
\setlength{\tabcolsep}{2pt}
\begin{tabularx}{\textwidth}{X""" + "c" * len(BASELINES) + r"""}
\toprule
\textbf{Metric} & """ + head + r""" \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabularx}
\end{table}
""")
    write_table(TAB / f"{tab_label.replace('tab:', 'tab_')}.tex", wide(tex))


def table_metric_matrix(df, group, gv, metrics, tab_label, caption):
    """Means of several metrics for every method at one condition."""
    sub = df[df[group] == gv]
    rows = []
    for m in METHODS:
        cells = []
        for metric in metrics:
            label, hb, d = METRICS[metric]
            mm, h, _ = mean_ci(sub[sub.method == m][metric])
            vals = {mm2: sub[sub.method == mm2][metric].mean() for mm2 in METHODS}
            best = (max if hb else min)(vals, key=vals.get)
            s_ = f"{mm:.{d}f}" if mm == mm else "--"
            if best == m:
                s_ = "\\textbf{" + s_ + "}"
            cells.append(s_)
        rows.append(f"{LABEL[m]} & " + " & ".join(cells) + " \\\\")
    head = " & ".join("\\textbf{" + METRICS[mm][0].replace("[", "\\newline[") + "}" for mm in metrics)
    tex = (r"""\begin{table}[H]
\caption{""" + caption + r"""\label{""" + tab_label + r"""}}
\footnotesize
\setlength{\tabcolsep}{2pt}
\begin{tabularx}{\textwidth}{l""" + "C" * len(metrics) + r"""}
\toprule
\textbf{Method} & """ + head + r""" \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabularx}
\end{table}
""")
    write_table(TAB / f"{tab_label.replace('tab:', 'tab_')}.tex", wide(tex))


def summarise(df, group, metrics, key):
    out = {}
    for gv, sub in df.groupby(group):
        out[str(gv)] = {}
        for m in METHODS:
            s = sub[sub.method == m]
            out[str(gv)][m] = {metric: mean_ci(s[metric])[:2] for metric in metrics if metric in s}
    SUMMARY[key] = out


def e1():
    df = load("E1")
    if df is None:
        return None, None
    metrics = list(METRICS)
    tests = all_tests(df, "lam", metrics)
    summarise(df, "lam", metrics, "E1")
    SUMMARY["E1_tests"] = tests.to_dict(orient="records")
    fr = []
    for metric in ("tdr_1", "gdr", "energy_per_reading_mJ", "delay_p95"):
        f = friedman(df, metric, "lam", higher_better=METRICS[metric][1])
        fr.append(f)
    SUMMARY["E1_friedman"] = pd.concat(fr).to_dict(orient="records")
    table_by_load(df, tests, "tdr_1", "tab:e1tdr",
                  "E1: timely delivery ratio (TDR) within 1\\,s, TDR(1\\,s); mean\\,$\\pm$\\,95\\% confidence interval (CI) over 15 test seeds, by offered load (readings/s per node). Bold: best mean. Markers compare GUARD with each baseline (Wilcoxon signed-rank, Holm over 8 baselines): $^{\\blacktriangle}$ GUARD significantly better, $^{\\triangledown}$ significantly worse, $^{\\circ}$ no significant difference.")
    table_by_load(df, tests, "gdr", "tab:e1gdr",
                  "E1: delivery ratio (delivered/generated readings); notation as in Table~\\ref{tab:e1tdr}.")
    table_by_load(df, tests, "energy_per_reading_mJ", "tab:e1energy",
                  "E1: energy per delivered reading [mJ]; notation as in Table~\\ref{tab:e1tdr} (lower is better; $^{\\blacktriangle}$: GUARD uses significantly less energy).")
    table_by_load(df, tests, "delay_p95", "tab:e1p95",
                  "E1: 95th-percentile delay of delivered readings [s]; notation as in Table~\\ref{tab:e1tdr} (lower is better).")
    table_wtl(tests, "lam", df.lam.nunique(), "tab:e1wtl",
              "E1: number of loads (out of 6) at which GUARD is significantly better / not significantly different / significantly worse than each baseline (Wilcoxon signed-rank, Holm over 8 baselines, $\\alpha=0.05$, 15 seeds). Italics: at least one significant loss.")
    for lam in (1.0, 3.0):
        table_metric_matrix(df, "lam", lam, ["tdr_05", "tdr_2", "delay_mean", "delay_p99", "jain",
                                             "coll_frac_all", "sink_coll_frac", "frames_per_reading", "ctrl_per_s"],
                            f"tab:e1sec{int(lam)}",
                            f"E1 at {lam:g} reading{'s' if lam != 1 else ''}/s per node: secondary metrics (means over 15 seeds; bold: best).")
    tests.to_csv(RES / "paired_tests_v2_E1.csv", index=False)
    return df, tests


def e2():
    df = load("E2")
    if df is None:
        return
    metrics = ["tdr_1", "gdr", "delay_p95", "energy_per_reading_mJ", "jain", "coll_frac_all"]
    tests = all_tests(df, "n", metrics)
    summarise(df, "n", metrics, "E2")
    SUMMARY["E2_tests"] = tests.to_dict(orient="records")
    table_by_load(df, tests, "tdr_1", "tab:e2tdr",
                  "E2 (scalability, 1 reading/s per node, constant density): TDR(1\\,s) by number of sensors; mean\\,$\\pm$\\,95\\% CI over 10 seeds; markers as in Table~\\ref{tab:e1tdr}.",
                  group="n", head_fmt="{:d}")
    table_by_load(df, tests, "energy_per_reading_mJ", "tab:e2energy",
                  "E2: energy per delivered reading [mJ] by number of sensors; markers as in Table~\\ref{tab:e1tdr}.",
                  group="n", head_fmt="{:d}")
    tests.to_csv(RES / "paired_tests_v2_E2.csv", index=False)


def _scen(df):
    return df["tag"].str.split(":").str[1]


def e3():
    df = load("E3")
    if df is None:
        return
    df["scen"] = _scen(df)
    metrics = ["tdr_1", "gdr", "delay_p95", "energy_per_reading_mJ", "jain", "gdr_crit", "delay_crit_p95"]
    tests = all_tests(df, "scen", metrics)
    summarise(df, "scen", metrics, "E3")
    SUMMARY["E3_tests"] = tests.to_dict(orient="records")
    order = ["corner", "clustered", "events", "cc2420", "harsh", "mild"]
    df["scen"] = pd.Categorical(df["scen"], order)
    METRICS["gdr_crit"] = ("delivery ratio of critical readings", True, 3)
    METRICS["delay_crit_p95"] = ("p95 delay of critical readings [s]", False, 3)
    table_by_load(df, tests, "tdr_1", "tab:e3tdr",
                  "E3 (robustness, 1 reading/s per node; events: 0.5 readings/s plus critical bursts): TDR(1\\,s), mean\\,$\\pm$\\,95\\% CI over 10 seeds. Corner: sink in a corner; clustered: Thomas deployment; CC2420: state-based energy model; harsh/mild: guard factor $\\Delta=1.0$/$0.3$. Markers as in Table~\\ref{tab:e1tdr}.",
                  group="scen", head_fmt="{}")
    table_by_load(df, tests, "energy_per_reading_mJ", "tab:e3energy",
                  "E3: energy per delivered reading [mJ] (the CC2420 model charges listening and therefore gives larger values); markers as in Table~\\ref{tab:e1tdr}.",
                  group="scen", head_fmt="{}")
    tests.to_csv(RES / "paired_tests_v2_E3.csv", index=False)


def e4():
    df = load("E4")
    if df is None:
        return
    df["fnd_f"] = df["fnd"].fillna(df["T"])
    df["hnd_f"] = df["hnd"].fillna(df["T"])
    METRICS.update({"fnd_f": ("first node dead [s]", True, 1), "hnd_f": ("half nodes dead [s]", True, 1),
                    "delivered": ("delivered readings", True, 0), "timely": ("timely readings", True, 0),
                    "alive_frac": ("alive fraction at 900 s", True, 3)})
    metrics = ["fnd_f", "hnd_f", "alive_frac", "delivered", "timely", "tdr_1", "gdr", "energy_per_reading_mJ"]
    df["grp"] = "E4"
    tests = all_tests(df, "grp", metrics)
    SUMMARY["E4_tests"] = tests.to_dict(orient="records")
    summarise(df, "grp", metrics, "E4")
    rows = []
    for m in METHODS:
        s = df[df.method == m]
        cells = []
        for metric in metrics:
            label, hb, d = METRICS[metric]
            mm, h, _ = mean_ci(s[metric])
            vals = {mm2: df[df.method == mm2][metric].mean() for mm2 in METHODS}
            best = (max if hb else min)(vals, key=vals.get)
            c = f"{mm:.{d}f}"
            if best == m:
                c = "\\textbf{" + c + "}"
            if m != "guard":
                sub = tests[(tests.metric == metric) & (tests.baseline == m)]
                c += sig_symbol(sub.iloc[0] if len(sub) else None, hb)
            cells.append(c)
        rows.append(f"{LABEL[m]} & " + " & ".join(cells) + " \\\\")
    head = " & ".join("\\textbf{" + METRICS[mm][0] + "}" for mm in metrics)
    tex = (r"""\begin{table}[H]
\caption{E4 (lifetime: $E_0=0.3$\,J, 1 reading/s per node, 900\,s): means over 10 seeds (death times: 900\,s if no such death occurred). Delivered and timely readings are totals over the measurement window. Markers as in Table~\ref{tab:e1tdr}.\label{tab:e4}}
\footnotesize
\setlength{\tabcolsep}{2pt}
\begin{tabularx}{\textwidth}{l""" + "C" * len(metrics) + r"""}
\toprule
\textbf{Method} & """ + head + r""" \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabularx}
\end{table}
""")
    write_table(TAB / "tab_e4.tex", wide(tex))
    # time series (mean over seeds)
    ts = {}
    for m in METHODS:
        s = df[df.method == m]
        al = [r["ts"]["alive"] for _, r in s.iterrows() if isinstance(r.get("ts"), dict)]
        dl = [r["ts"]["delivered"] for _, r in s.iterrows() if isinstance(r.get("ts"), dict)]
        if al:
            L = min(len(a) for a in al)
            ts[m] = {"alive": np.mean([a[:L] for a in al], axis=0).tolist(),
                     "delivered": np.mean([d_[:L] for d_ in dl], axis=0).tolist()}
    SUMMARY["E4_ts"] = ts
    # post hoc (not pre-registered): time until the 20-s moving average of delivered
    # readings first falls below 90 % / 50 % of the offered load (after 60 s).
    # Reported because no method reached the pre-registered half-nodes-dead time.
    def collapse(ts_, thr, k=20):
        d = np.array(ts_["delivered"], dtype=float)
        ma = np.convolve(d, np.ones(k) / k, mode="valid")
        t = np.arange(k, len(d) + 1)
        idx = np.where((ma < thr) & (t > 60))[0]
        return float(t[idx[0]]) if len(idx) else float(len(d))
    offered = float(df["lam"].iloc[0] * df["n"].iloc[0])
    df["collapse90"] = [collapse(r, 0.9 * offered) for r in df["ts"]]
    df["collapse50"] = [collapse(r, 0.5 * offered) for r in df["ts"]]
    ct = all_tests(df, "grp", ["collapse90", "collapse50"])
    SUMMARY["E4_collapse"] = {m: {k: mean_ci(df[df.method == m][k])[:2] for k in ("collapse90", "collapse50")}
                              for m in METHODS}
    SUMMARY["E4_collapse_tests"] = ct.to_dict(orient="records")
    tests = pd.concat([tests, ct], ignore_index=True)
    tests.to_csv(RES / "paired_tests_v2_E4.csv", index=False)


def e5():
    df = load("E5")
    if df is None:
        return
    df["variant"] = df["tag"].str.split(":").str[1]
    metrics = ["tdr_1", "gdr", "delay_mean", "delay_p95", "energy_per_reading_mJ", "coll_frac_all",
               "frames_per_reading"]
    order = ["full", "no_dispatch", "no_nlr", "with_decorrelation", "no_edf", "static_etx",
             "no_energy_terms", "model_prior", "ctp_like"]
    names = {"full": "GUARD (tuned)", "no_dispatch": "-- priced dispatch", "no_nlr": "-- network-layer retransmission",
             "with_decorrelation": "+ decorrelation delay ($b$ = 10 ms)", "no_edf": "FIFO instead of EDF",
             "static_etx": "static ETX instead of measured", "no_energy_terms": "-- energy terms",
             "model_prior": "+ hidden-terminal model prior",
             "ctp_like": "collection-tree core (measured ETX and retransmission only)"}
    res = {}
    loads = sorted(df.lam.unique())
    from scipy import stats as st

    def cellv(v, metric, lam):
        sl = df[(df.variant == v) & (df.lam == lam)].set_index("seed")[metric]
        fl = df[(df.variant == "full") & (df.lam == lam)].set_index("seed")[metric]
        mm = sl.mean()
        d = 2 if metric == "frames_per_reading" else 3
        c = f"{mm:.{d}f}"
        if v != "full":
            pair = pd.concat([fl, sl], axis=1, keys=["f", "v"]).dropna()
            dd = (pair.f - pair.v).to_numpy()
            p = st.wilcoxon(dd).pvalue if np.any(dd != 0) else 1.0
            res.setdefault(v, {}).setdefault(str(lam), {})[f"p_{metric}"] = float(p)
            c += "$^{*}$" if p < 0.05 / 8 else ""
        res.setdefault(v, {}).setdefault(str(lam), {})[metric] = mean_ci(sl)[:2]
        return c

    def build(metrics, labels, tab_label, caption):
        rows = []
        for v in order:
            cells = [cellv(v, mtr, lam) for mtr in metrics for lam in loads]
            rows.append(f"{names[v]} & " + " & ".join(cells) + " \\\\")
        head1 = "".join(" & \\multicolumn{%d}{c}{\\textbf{%s}}" % (len(loads), lab) for lab in labels)
        head2 = " & ".join([f"{lam:g}" for lam in loads] * len(metrics))
        tex = (r"""\begin{table}[H]
\caption{""" + caption + r"""\label{""" + tab_label + r"""}}
\footnotesize
\setlength{\tabcolsep}{2pt}
\begin{tabularx}{\textwidth}{X""" + "c" * (len(metrics) * len(loads)) + r"""}
\toprule
\textbf{Variant}""" + head1 + r""" \\
 & """ + head2 + r""" \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabularx}
\end{table}
""")
        write_table(TAB / f"{tab_label.replace('tab:', 'tab_')}.tex", wide(tex))

    for v in order:
        for lam in loads:
            for mtr in metrics:
                cellv(v, mtr, lam)
    build(["tdr_1", "energy_per_reading_mJ"], ["TDR(1\\,s)", "energy per reading [mJ]"], "tab:e5",
          "E5: ablation of GUARD, timely delivery and energy (means over 10 seeds; columns: offered load in readings/s per node). Each row removes, adds or replaces one component of the tuned configuration; FIFO: first in, first out. $^{*}$: different from the full method (two-sided Wilcoxon signed-rank test, Bonferroni over the 8 variants, $\\alpha=0.05$).")
    build(["delay_p95", "frames_per_reading"], ["95th-percentile delay [s]", "frames per delivered reading"], "tab:e5b",
          "E5: ablation of GUARD, tail delay and channel use; notation as in Table~\\ref{tab:e5}.")
    SUMMARY["E5"] = res


E6_METRICS = ("tdr_1", "gdr", "delay_mean", "delay_p95", "energy_per_reading_mJ", "frames_per_reading")


def _sweep_tests(df, key, ref_value, metrics):
    """Paired Wilcoxon tests of every swept value against the tuned value (Holm over the
    values of one sweep).  df must contain one sweep (column `key`)."""
    d = df.copy()
    d["method"] = d[key].astype(str)
    d["grp"] = "sweep"
    ref = str(ref_value)
    vals = sorted(d["method"].unique(), key=float)
    out = []
    for m in metrics:
        t = paired_tests(d, m, "grp", ref=ref, methods=vals)
        if len(t):
            out.append(t)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def e6():
    df = load("E6")
    if df is None:
        return
    df["param"] = df["tag"].str.split(":").str[1].str.split("=").str[0]
    df["value"] = df["tag"].str.split("=").str[1].astype(float)
    tuned = json.loads((ROOT / "results" / "tuned_params.json").read_text())["guard"]["params"]
    out, tests = {}, {}
    for (p, v), s in df.groupby(["param", "value"]):
        out.setdefault(p, {})[str(v)] = {mm: mean_ci(s[mm])[:2] for mm in E6_METRICS}
    for p, s in df.groupby("param"):
        ref = float(tuned[p])
        t = _sweep_tests(s.assign(value=s["value"].astype(float)), "value", ref, E6_METRICS)
        t["param"] = p
        tests[p] = t.to_dict(orient="records")
    SUMMARY["E6"] = out
    SUMMARY["E6_tuned"] = {p: float(tuned[p]) for p in out}
    SUMMARY["E6_tests"] = tests


def e7():
    df = load("E7")
    if df is None:
        return
    df["h"] = df["tag"].str.split("=").str[1]
    out = {}
    for (h, lam), s in df.groupby(["h", "lam"]):
        out.setdefault(h, {})[str(lam)] = {mm: mean_ci(s[mm])[:2] for mm in
                                           ("tdr_05", "tdr_1", "gdr", "delay_mean", "delay_p95",
                                            "energy_per_reading_mJ", "frames_per_reading")}
    SUMMARY["E7"] = out
    # every h against the tuned h = 0.01, per load ("inf" = no holding at all)
    tests = {}
    for lam, s in df.groupby("lam"):
        d = s.copy()
        d["hv"] = d["h"].replace({"inf": "1e9"}).astype(float)
        t = _sweep_tests(d, "hv", 0.01, ("tdr_05", "tdr_1", "delay_mean", "delay_p95",
                                         "energy_per_reading_mJ"))
        tests[str(lam)] = t.to_dict(orient="records")
    SUMMARY["E7_tests"] = tests


def e8():
    df = load("E8")
    if df is None:
        return
    metrics = ["tdr_1", "gdr", "delay_p95", "energy_per_reading_mJ"]
    tests = all_tests(df, "lam", metrics)
    summarise(df, "lam", metrics, "E8")
    SUMMARY["E8_tests"] = tests.to_dict(orient="records")
    table_by_load(df, tests, "tdr_1", "tab:e8tdr",
                  "E8: TDR(1\\,s) of the default (untuned) configuration of every method, mean\\,$\\pm$\\,95\\% CI over 10 seeds; markers compare default GUARD with each default baseline as in Table~\\ref{tab:e1tdr}.")
    tests.to_csv(RES / "paired_tests_v2_E8.csv", index=False)


def complexity(df):
    if df is None:
        return
    rows = []
    out = {}
    rt_path = RES / "route_timing.json"
    rt = json.loads(rt_path.read_text()) if rt_path.exists() else None
    for m in METHODS:
        s = df[df.method == m]
        dec = mean_ci(s["dec_us"])[0]
        sink = mean_ci(s["sink_ms_per_s"])[0]
        if m == "guard" and rt is not None:
            sink = rt["100"]["ms_per_update"]          # one route update per simulated second
        ctrl = mean_ci(s["ctrl_per_s"])[0]
        hdr = {"guard": 0, "twinloop": 2, "bcp": 2, "cbpr": 2}.get(m, 0)
        out[m] = {"dec_us": dec, "sink_ms_per_s": sink, "ctrl_per_s": ctrl, "hdr_bytes": hdr}
        ovh = mean_ci(s["ovh_per_s"])[0]
        out[m]["ovh_per_s"] = ovh
        rows.append(f"{LABEL[m]} & {dec:.1f} & {sink:.2f} & {ctrl:.2f} & {hdr} & {ovh:.2f} \\\\")
    SUMMARY["complexity"] = out
    tex = (r"""\begin{table}[H]
\caption{E1: computation and overhead, averaged over all loads and seeds. Decision time: wall-clock time of the simulator per forwarding decision at a node; central time: computation per simulated second for cluster formation at the sink (clustered methods) or for GUARD's centrally emulated route update; control: bytes of beacons, beacon extensions and cluster-formation messages; extra header: bytes added to every data frame; overhead: control plus extra-header bytes.\label{tab:complexity}}
\footnotesize
\begin{tabularx}{\textwidth}{lCCCCC}
\toprule
\textbf{Method} & \textbf{Decision time [\textmu{}s]} & \textbf{Central time [ms/s]} & \textbf{Control [B/(node\,s)]} & \textbf{Extra header [B]} & \textbf{Overhead [B/(node\,s)]}\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabularx}
\end{table}
""")
    write_table(TAB / "tab_complexity.tex", tex)


# -------------------------------------------------------------------- figures
def _lines(ax, df, metric, group="lam", methods=PLOT, ci=True):
    for m in methods:
        s = df[df.method == m]
        xs, ys, hs = [], [], []
        for gv in sorted(s[group].unique()):
            mm, h, _ = mean_ci(s[s[group] == gv][metric])
            xs.append(gv); ys.append(mm); hs.append(h if h == h else 0.0)
        ax.plot(xs, ys, marker=MARKER[m], color=COLOR[m], label=LABEL[m], lw=2.0 if m == "guard" else 1.3,
                ms=5 if m == "guard" else 4, zorder=3 if m == "guard" else 2,
                markeredgecolor="white", markeredgewidth=0.6)
        if ci:
            ax.fill_between(xs, np.array(ys) - hs, np.array(ys) + hs, color=COLOR[m], alpha=0.12, lw=0)


def fig_e1(df):
    import matplotlib.pyplot as plt
    setup_mpl()
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.4))
    for ax, metric, title in zip(axs, ["tdr_1", "gdr", "tdr_05"], ["(a) TDR(1 s)", "(b) delivery ratio", "(c) TDR(0.5 s)"]):
        _lines(ax, df, metric)
        ax.set_title(title, loc="left")
        ax.set_xlabel("offered load [readings/s per node]")
        ax.set_ylim(0, 1.02)
    axs[0].set_ylabel("ratio")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=7, frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    _save(fig, "fig_e1_delivery")
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.4))
    for ax, metric, title, log in zip(axs, ["delay_mean", "delay_p95", "energy_per_reading_mJ"],
                                      ["(a) mean delay [s]", "(b) 95th-percentile delay [s]", "(c) energy per reading [mJ]"],
                                      [True, True, False]):
        _lines(ax, df, metric, ci=False)
        ax.set_title(title, loc="left")
        ax.set_xlabel("offered load [readings/s per node]")
        if log:
            ax.set_yscale("log")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=7, frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    _save(fig, "fig_e1_delay_energy")
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.4))
    for ax, metric, title in zip(axs, ["sink_coll_frac", "frames_per_reading", "jain"],
                                 ["(a) collided frames at the sink", "(b) frames per delivered reading",
                                  "(c) Jain fairness index"]):
        _lines(ax, df, metric, ci=False)
        ax.set_title(title, loc="left")
        ax.set_xlabel("offered load [readings/s per node]")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=7, frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    _save(fig, "fig_e1_mechanism")


def fig_e2():
    df = load("E2")
    if df is None:
        return
    import matplotlib.pyplot as plt
    setup_mpl()
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.4))
    for ax, metric, title in zip(axs, ["tdr_1", "delay_p95", "energy_per_reading_mJ"],
                                 ["(a) TDR(1 s)", "(b) 95th-percentile delay [s]", "(c) energy per reading [mJ]"]):
        _lines(ax, df, metric, group="n", ci=False)
        ax.set_title(title, loc="left")
        ax.set_xticks([50, 100, 200, 300])
        ax.set_xlabel("number of sensors")
        if metric == "delay_p95":
            ax.set_yscale("log")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=7, frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    _save(fig, "fig_e2_scalability")


def fig_e3():
    df = load("E3")
    if df is None:
        return
    import matplotlib.pyplot as plt
    setup_mpl()
    df["scen"] = _scen(df)
    order = ["corner", "clustered", "events", "cc2420", "harsh", "mild"]
    ticks = ["sink in\ncorner", "clustered\ndeployment", "event\nbursts", "CC2420\nenergy",
             "$\\Delta=1.0$\n(harsh)", "$\\Delta=0.3$\n(mild)"]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7))
    w = 0.8 / len(PLOT)
    ref = {sc: df[(df.method == "guard") & (df.scen == sc)]["energy_per_reading_mJ"].mean() for sc in order}
    for ax, metric, title in zip(axs, ["tdr_1", "energy_rel"],
                                 ["(a) TDR(1 s)", "(b) energy per reading relative to GUARD"]):
        for k, m in enumerate(PLOT):
            if metric == "energy_rel":
                ys = [df[(df.method == m) & (df.scen == sc)]["energy_per_reading_mJ"].mean() / ref[sc]
                      for sc in order]
            else:
                ys = [df[(df.method == m) & (df.scen == sc)][metric].mean() for sc in order]
            ax.bar(np.arange(len(order)) + (k - len(PLOT) / 2 + 0.5) * w, ys, width=w,
                   color=COLOR[m], label=LABEL[m], edgecolor="white", linewidth=0.8, zorder=2)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(ticks, fontsize=6.5)
        ax.grid(axis="x", visible=False)
        ax.set_title(title, loc="left")
        if metric == "energy_rel":
            ax.axhline(1.0, color=TEXT2, lw=0.8, ls=(0, (3, 2)), zorder=3)
            ax.set_ylim(0, None)
        else:
            ax.set_ylim(0, 1.0)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=7, frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    _save(fig, "fig_e3_robustness")


def fig_e4():
    ts = SUMMARY.get("E4_ts")
    if not ts:
        return
    import matplotlib.pyplot as plt
    setup_mpl()
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.5))
    for m in PLOT:
        if m not in ts:
            continue
        al = np.array(ts[m]["alive"])
        dl = np.array(ts[m]["delivered"])
        t = np.arange(1, len(al) + 1)
        k = 20
        sm = np.convolve(dl, np.ones(k) / k, mode="valid")
        axs[0].plot(t, al, color=COLOR[m], label=LABEL[m], lw=2.0 if m == "guard" else 1.2)
        axs[1].plot(t[k - 1:], sm, color=COLOR[m], label=LABEL[m], lw=2.0 if m == "guard" else 1.2)
    axs[0].set_title("(a) alive sensors", loc="left")
    axs[1].set_title("(b) delivered readings per second (20-s moving average)", loc="left")
    for ax in axs:
        ax.set_xlabel("time [s]")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=7, frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    _save(fig, "fig_e4_lifetime")


def fig_e7(e1df):
    """E7 as small multiples of horizontal bars: one row per method or GUARD operating point.

    Columns: 95th-percentile delay of delivered readings, energy per delivered reading and the
    share of readings delivered within 1 s; rows of panels: 1 and 2 readings/s per node. GUARD is
    shown at three holding costs (no holding, tuned h = 0.01, h = 0.0025); the baselines are the
    tuned methods of E1 on the same 10 seeds. Whiskers: 95% t-confidence intervals. Drawn at its
    printed size (full page width), so the text prints at 7-8 pt.
    """
    e7 = SUMMARY.get("E7")
    if not e7 or e1df is None:
        return
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator, StrMethodFormatter
    setup_mpl()
    seeds = sorted(e1df.seed.unique())[:10]
    guard_rows = [("inf", "GUARD, no holding"), ("0.01", "GUARD, $h$ = 0.01 (tuned)"), ("0.0025", "GUARD, $h$ = 0.0025")]
    cols = [("delay_p95", "95th-percentile delay [ms]", 1000.0, 500.0, 100, "{:.0f}"),
            ("energy_per_reading_mJ", "energy per delivered reading [mJ]", 1.0, 1.0, 0.2, "{:.3f}"),
            ("tdr_1", "readings delivered within 1 s [%]", 100.0, 100.0, 20, "{:.1f}")]
    labels = [lab for _, lab in guard_rows] + [LABEL[m] for m in BASELINES]
    n = len(labels)
    ypos = list(range(n))
    ypos = [y + (0.6 if i >= len(guard_rows) else 0.0) for i, y in enumerate(ypos)]   # gap after GUARD rows
    GRAY = "#a3acb6"
    fig, axs = plt.subplots(2, 3, figsize=(7.27, 4.7), sharey=True,
                            gridspec_kw=dict(wspace=0.2, hspace=0.45))
    for r, lam in enumerate((1.0, 2.0)):
        for c, (metric, title, scale, xmax, step, fmt) in enumerate(cols):
            ax = axs[r][c]
            vals = []
            for h, _ in guard_rows:
                mean, ci = e7[h][str(lam)][metric]
                vals.append((scale * mean, scale * ci, COLOR["guard"]))
            for m in BASELINES:
                sel = e1df[(e1df.method == m) & (e1df.lam == lam) & (e1df.seed.isin(seeds))]
                mean, ci, _ = mean_ci(scale * sel[metric])
                vals.append((mean, ci, GRAY))
            for y, (v, ci, col) in zip(ypos, vals):
                shown = min(v, xmax)
                ax.barh(y, shown, height=0.66, color=col, edgecolor="none", zorder=2)
                if v > xmax:      # off scale: break mark and the true value
                    ax.plot([xmax * 0.955, xmax * 0.975], [y - 0.42, y + 0.42], color="white", lw=2.4, zorder=3)
                    ax.text(xmax * 0.94, y, f"{v / 1000:.1f} s", ha="right", va="center", fontsize=6.6,
                            color=TEXT, fontweight="bold", zorder=4)
                    continue
                if ci == ci and ci > 0:
                    ax.errorbar(v, y, xerr=ci, fmt="none", ecolor="#3d4852", elinewidth=0.7, capsize=1.6,
                                capthick=0.7, zorder=3)
                end = v + (ci if ci == ci else 0)
                txt = fmt.format(v)
                if end + 0.02 * xmax > xmax * 0.86:      # no room on the right: write it inside the bar
                    ax.text(v - (ci if ci == ci else 0) - 0.015 * xmax, y, txt, ha="right", va="center",
                            fontsize=6.6, color="white" if col == COLOR["guard"] else TEXT, zorder=4)
                else:
                    ax.text(end + 0.015 * xmax, y, txt, ha="left", va="center", fontsize=6.6, color=TEXT, zorder=4)
            ax.set_xlim(0, xmax)
            ax.xaxis.set_major_locator(MultipleLocator(step))
            ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.1f}" if metric == "energy_per_reading_mJ" else "{x:.0f}"))
            ax.grid(True, axis="x", color=GRID, lw=0.6)
            ax.grid(False, axis="y")
            ax.set_axisbelow(True)
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", length=0)
            ax.axhline((ypos[len(guard_rows) - 1] + ypos[len(guard_rows)]) / 2, color="#c8ced4", lw=0.6, zorder=1)
            ax.set_xlabel(title, fontsize=7.6)
            if c == 0:
                ax.set_title(f"$\\bf{{({'a' if lam == 1 else 'b'})}}$  {lam:g} reading{'s' if lam != 1 else ''}/s per node",
                             loc="left", fontsize=8.5, pad=5, x=-0.52)
    axs[0][0].set_yticks(ypos)
    axs[0][0].set_yticklabels(labels, fontsize=7.3)
    for row in axs:
        for lab in row[0].get_yticklabels():
            lab.set_color(TEXT)
    for lab in axs[0][0].get_yticklabels() + axs[1][0].get_yticklabels():
        if "tuned" in lab.get_text():
            lab.set_fontweight("bold")
    axs[0][0].invert_yaxis()
    fig.subplots_adjust(left=0.205, right=0.985, top=0.94, bottom=0.085)
    _save(fig, "fig_e7_frontier")


def fig_e5_e6():
    import matplotlib.pyplot as plt
    setup_mpl()
    e6 = SUMMARY.get("E6")
    if not e6:
        return
    tuned = SUMMARY.get("E6_tuned", {})
    tests = SUMMARY.get("E6_tests", {})
    params = [p for p in ["kappa", "beta", "alpha_s", "nlr", "rdel", "u0"] if p in e6]
    names = {"kappa": r"$\kappa$", "beta": r"$\beta$", "alpha_s": r"$\alpha$", "nlr": r"$N_\mathrm{r}$",
             "rdel": r"$b$ [ms]", "u0": r"$u_0$"}
    rows = [("tdr_1", "TDR(1 s)", 1.0), ("delay_p95", "p95 delay [ms]", 1000.0),
            ("energy_per_reading_mJ", "energy [mJ]", 1.0)]
    fig, axs = plt.subplots(len(rows), len(params), figsize=(7.2, 4.6), sharey="row")
    for c, p in enumerate(params):
        vs = sorted(e6[p], key=float)
        xscale = 1000.0 if p == "rdel" else 1.0
        x = [float(v) * xscale for v in vs]
        for r, (m, lab, ys) in enumerate(rows):
            ax = axs[r][c]
            y = [e6[p][v][m][0] * ys for v in vs]
            h = [e6[p][v][m][1] * ys for v in vs]
            sig = {}
            for t in tests.get(p, []):
                if t["metric"] == m:
                    sig[float(t["baseline"])] = t["p_holm"] < 0.05
            ax.errorbar(x, y, yerr=h, color=COLOR["guard"], lw=1.5, ms=0, capsize=1.5, zorder=2)
            for xv, yv, v in zip(x, y, vs):
                is_ref = abs(float(v) - tuned.get(p, -1)) < 1e-12
                filled = is_ref or sig.get(float(v), False)
                ax.plot([xv], [yv], marker="D" if is_ref else "o", ms=4.5 if is_ref else 3.5,
                        mfc=COLOR["guard"] if filled else "white", mec=COLOR["guard"], mew=1.0,
                        ls="none", zorder=3)
            if p in ("kappa", "alpha_s"):
                ax.set_xscale("log", base=2)
            tick_sel = {"beta": [0.05, 0.25, 0.5, 0.75], "rdel": [0.0, 10.0, 20.0, 50.0]}
            xt = tick_sel.get(p, x)
            ax.set_xticks(xt)
            ax.set_xticklabels([f"{xv:g}" for xv in xt] if r == len(rows) - 1 else [], fontsize=6.5)
            ax.minorticks_off()
            if r == len(rows) - 1:
                ax.set_xlabel(names[p])
            if c == 0:
                ax.set_ylabel(lab)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="D", ls="none", mfc=COLOR["guard"], mec=COLOR["guard"], label="tuned value"),
               Line2D([], [], marker="o", ls="none", mfc=COLOR["guard"], mec=COLOR["guard"],
                      label="significantly different from tuned value (Holm)"),
               Line2D([], [], marker="o", ls="none", mfc="white", mec=COLOR["guard"], label="not significant")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    _save(fig, "fig_e6_sensitivity")


def _save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=600)
    import matplotlib.pyplot as plt
    plt.close(fig)


def main():
    what = set(sys.argv[1:]) or {"tables", "figures"}
    sys.path.insert(0, str(ROOT / "src"))
    if (ROOT / "results" / "tuned_params.json").exists():
        table_tuned()
        table_guard_params()
    e1df, _ = e1()
    complexity(e1df)
    e2(); e3(); e4(); e5(); e6(); e7(); e8()
    per_metric_tables()
    if "figures" in what:
        if e1df is not None:
            fig_e1(e1df)
        fig_e2(); fig_e3(); fig_e4(); fig_e7(e1df); fig_e5_e6()
    (RES / "summary_v2.json").write_text(json.dumps(SUMMARY, indent=1, default=float))
    print("summary keys:", list(SUMMARY))


if __name__ == "__main__":
    main()
