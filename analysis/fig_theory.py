"""Figure of the hidden guard zone (Lemma 1, Equation (lens)): geometry and area.

Usage: python analysis/fig_theory.py   (needs results/theory/theory_checks.json)
"""
from __future__ import annotations

import json
import math

import numpy as np

from common import COLOR, FIG, ROOT, TEXT, TEXT2, setup_mpl
from theory_checks import hidden_area


def main():
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    setup_mpl()
    chk = json.loads((ROOT / "results" / "theory" / "theory_checks.json").read_text())
    delta, rcs = 0.6, 150.0
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1, 1.25]})
    # (a) geometry for d = 90 m
    d = 90.0
    a = (1 + delta) * d
    xi, xj = (0.0, 0.0), (d, 0.0)
    gx = np.linspace(xj[0] - a, xj[0] + a, 700)
    gy = np.linspace(-a, a, 700)
    X, Y = np.meshgrid(gx, gy)
    inz = (X - xj[0]) ** 2 + Y ** 2 < a * a
    hid = inz & (X ** 2 + Y ** 2 > rcs ** 2)
    a1.contourf(X, Y, hid.astype(float), levels=[0.5, 1.5], colors=[COLOR["guard"]], alpha=0.35)
    a1.add_patch(Circle(xj, a, fill=False, ec=COLOR["guard"], lw=1.2))
    a1.add_patch(Circle(xi, rcs, fill=False, ec=TEXT2, lw=1.0, ls="--"))
    a1.plot(*xi, "o", color=TEXT, ms=4)
    a1.plot(*xj, "s", color=TEXT, ms=4)
    a1.annotate("", xy=xj, xytext=xi, arrowprops=dict(arrowstyle="-|>", color=TEXT, lw=1.0))
    a1.text(-12, -16, "$i$", color=TEXT)
    a1.text(d + 4, -16, "$j$", color=TEXT)
    a1.text(d / 2 - 8, 6, "$d_{ij}$", color=TEXT, fontsize=7)
    a1.text(-150, -172, "carrier sense of $i$ ($R_\\mathrm{cs}=150$ m)", color=TEXT2, fontsize=6.5)
    a1.text(d, a + 8, "guard zone, radius $(1+\\Delta)d_{ij}$", color=COLOR["guard"], fontsize=6.5, ha="center")
    a1.text(185, -10, "hidden\nzone", color=COLOR["guard"], fontsize=7, ha="center")
    a1.set_xlim(-160, d + a + 10)
    a1.set_ylim(-180, a + 24)
    a1.set_aspect("equal")
    a1.grid(False)
    a1.set_xticks([])
    a1.set_yticks([])
    for sp in a1.spines.values():
        sp.set_visible(False)
    a1.set_title(f"(a) link of {d:.0f} m, $\\Delta$ = 0.6", loc="left")
    # (b) area vs link length
    ds = np.linspace(0, 125, 400)
    ar = [hidden_area(x, delta, rcs) / 1e3 for x in ds]
    a2.plot(ds, ar, color=COLOR["guard"], lw=1.8, label="analytic (lens formula)")
    pts = chk["lemma1"]
    a2.plot([p["d"] for p in pts], [p["mc"] / 1e3 for p in pts], "o", color=COLOR["twinloop"], ms=3.5,
            label="Monte Carlo", markeredgecolor="white", markeredgewidth=0.5)
    thr = rcs / (2 + delta)
    a2.axvline(thr, color=TEXT2, lw=0.8, ls=":")
    a2.text(thr - 2, 55, f"$R_\\mathrm{{cs}}/(2+\\Delta)$ = {thr:.1f} m", rotation=90, ha="right", va="top",
            color=TEXT2, fontsize=6.5)
    a2.set_xlabel("link length $d_{ij}$ [m]")
    a2.set_ylabel("hidden area [10$^3$ m$^2$]")
    a2.legend(frameon=False, loc="upper left")
    a2.set_title("(b) area of the hidden guard zone", loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "fig_hidden_area.pdf")
    fig.savefig(FIG / "fig_hidden_area.png", dpi=600)
    print("written", FIG / "fig_hidden_area.pdf")


if __name__ == "__main__":
    main()
