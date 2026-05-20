"""
make_fig_mismatch.py
--------------------
Generate figures/fig_mismatch_robustness.pdf with two side-by-side subplots:

  Left:  bias sweep  (x = bias_m, y = empirical CVaR_beta)
  Right: cov underestimate (x = factor, log scale)

Each subplot overlays SensLyap (red) vs DRCVaRLyap (blue) with 95% bootstrap
CI shading from the per-seed list in exp_mismatch.json.
"""
from __future__ import annotations
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results", "exp_mismatch.json")
FIG_OUT = os.path.normpath(
    os.path.join(HERE, "..", "figures", "fig_mismatch_robustness.pdf"))


def _series(d: dict, knob_keys, ctl: str, metric: str):
    """Return (xs, means, los, his) for a controller across knob values."""
    xs, means, los, his = [], [], [], []
    for kkey, x in knob_keys:
        cell = d.get(kkey, {}).get(ctl, {}).get(metric, {})
        if not cell:
            continue
        xs.append(x)
        means.append(cell["mean"])
        los.append(cell["lo"])
        his.append(cell["hi"])
    return np.array(xs), np.array(means), np.array(los), np.array(his)


def main():
    with open(RESULTS) as f:
        d = json.load(f)

    bias_values = d["_meta"]["bias_values_m"]
    factor_values = d["_meta"]["factor_values"]
    bias_keys = [(f"bias={float(b)}", float(b)) for b in bias_values]
    fac_keys = [(f"factor={float(f)}", float(f)) for f in factor_values]

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
    })

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.0, 2.7))

    COL = {"SensLyap": "#d62728", "DRCVaRLyap": "#1f77b4"}
    LBL = {"SensLyap": "Sens-Lyap (no DR)",
           "DRCVaRLyap": "DR-CVaR (proposed)"}

    # ---- LEFT: bias sweep ----
    for ctl in ("SensLyap", "DRCVaRLyap"):
        x, m, lo, hi = _series(d, bias_keys, ctl, "cvar_beta")
        axL.plot(x, m, marker="o", lw=1.6, color=COL[ctl], label=LBL[ctl])
        axL.fill_between(x, lo, hi, color=COL[ctl], alpha=0.18, linewidth=0)
    axL.set_xlabel(r"position bias $b$ (m)")
    axL.set_ylabel(r"empirical CVaR$_{\beta=0.95}$")
    axL.set_xticks(bias_values)
    axL.grid(alpha=0.3)
    axL.legend(framealpha=0.5, loc="upper left")

    # ---- RIGHT: cov underestimate ----
    for ctl in ("SensLyap", "DRCVaRLyap"):
        x, m, lo, hi = _series(d, fac_keys, ctl, "cvar_beta")
        axR.plot(x, m, marker="s", lw=1.6, color=COL[ctl], label=LBL[ctl])
        axR.fill_between(x, lo, hi, color=COL[ctl], alpha=0.18, linewidth=0)
    axR.set_xscale("log", base=2)
    axR.set_xticks(factor_values)
    axR.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axR.set_xlabel(r"cov.\ underestimate factor $\sigma_{\rm true}/\sigma_{\rm rep}$")
    axR.set_ylabel(r"empirical CVaR$_{\beta=0.95}$")
    axR.grid(alpha=0.3, which="both")
    axR.legend(framealpha=0.5, loc="upper left")

    fig.tight_layout()
    os.makedirs(os.path.dirname(FIG_OUT), exist_ok=True)
    fig.savefig(FIG_OUT, format="pdf", bbox_inches="tight")
    print(f"wrote {FIG_OUT}")


if __name__ == "__main__":
    main()
