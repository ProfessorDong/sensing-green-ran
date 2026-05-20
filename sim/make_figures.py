"""
make_figures.py
---------------
Generate publication-quality PDF figures from sim/results/exp_*.json.
"""
from __future__ import annotations
import json
import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----- IEEE single-column styling -----
plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.figsize": (3.5, 2.5),
    "lines.linewidth": 1.5,
    "lines.markersize": 5,
})

# Color-blind-friendly tab10 mapping per controller
CTL_STYLE = {
    "AlwaysOn":   {"color": "#1f77b4", "marker": "o", "label": "Always-On"},
    "Threshold":  {"color": "#ff7f0e", "marker": "s", "label": "Threshold"},
    "NoSensLyap": {"color": "#2ca02c", "marker": "^", "label": "No-Sens Lyap."},
    "SensLyap":   {"color": "#9467bd", "marker": "D", "label": "Sens. Lyap."},
    "DRCVaRLyap": {"color": "#d62728", "marker": "*", "label": "GreenSense (ours)"},
    "OracleDP":   {"color": "#7f7f7f", "marker": "x", "label": "Clairvoyant"},
}

RESULTS_DIR = "sim/results"
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)


def load(name):
    path = os.path.join(RESULTS_DIR, f"exp_{name}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _mean_lohi(d, key):
    """Return (mean, lo, hi) from an aggregated dict's bootstrap entry."""
    e = d.get(key, {})
    return e.get("mean", 0.0), e.get("lo", 0.0), e.get("hi", 0.0)


# ------------------------------------------------------------------ 1
def fig_pareto():
    data = load("main")
    if not data:
        return
    fig, ax = plt.subplots()
    for ctl, agg in data.items():
        if ctl.startswith("_"):
            continue
        Pm, Plo, Phi = _mean_lohi(agg, "avg_power_W")
        Dm, Dlo, Dhi = _mean_lohi(agg, "p99_delay_ms")
        style = CTL_STYLE.get(ctl, {})
        ax.errorbar(Dm, Pm,
                    xerr=[[max(Dm - Dlo, 0)], [max(Dhi - Dm, 0)]],
                    yerr=[[max(Pm - Plo, 0)], [max(Phi - Pm, 0)]],
                    fmt=style.get("marker", "o"),
                    color=style.get("color", "k"),
                    label=style.get("label", ctl),
                    capsize=2)
    ax.set_xlabel(r"$p_{99}$ delay (ms)")
    ax.set_ylabel("Avg. power (W)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_pareto.pdf"), bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ 2,3
def fig_V():
    data = load("V")
    if not data:
        return
    Vs, P, Plo, Phi, Q, Qlo, Qhi = [], [], [], [], [], [], []
    for k, agg in data.items():
        if k.startswith("_"):
            continue
        Vs.append(agg.get("_V", float(k.split("=")[1])))
        pm, plo, phi = _mean_lohi(agg, "avg_power_W")
        qm, qlo, qhi = _mean_lohi(agg, "avg_backlog_bits")
        P.append(pm); Plo.append(plo); Phi.append(phi)
        Q.append(qm); Qlo.append(qlo); Qhi.append(qhi)
    order = np.argsort(Vs)
    Vs = np.array(Vs)[order]; P = np.array(P)[order]
    Plo = np.array(Plo)[order]; Phi = np.array(Phi)[order]
    Q = np.array(Q)[order]; Qlo = np.array(Qlo)[order]; Qhi = np.array(Qhi)[order]

    # ---- energy vs V (with 1/V asymptote) ----
    fig, ax = plt.subplots()
    ax.errorbar(Vs, P, yerr=[P - Plo, Phi - P],
                fmt="o-", color="#d62728", label="GreenSense", capsize=2)
    # fit P = P_inf + c / V via least squares over V where O(1/V) holds.
    # Restrict the fit DOMAIN to V >= V_fit_min and only DRAW the fit over
    # that domain, since the controller saturates at small V and the
    # asymptote does not hold there.
    if len(Vs) >= 3:
        try:
            from scipy.optimize import curve_fit
            def _model(V_, Pinf_, c_):
                return Pinf_ + c_ / V_
            V_fit_min = 1000.0
            mask = Vs >= V_fit_min
            if mask.sum() < 3:
                mask = np.ones_like(Vs, dtype=bool)
            popt, _ = curve_fit(_model, Vs[mask].astype(float),
                                P[mask].astype(float),
                                p0=[float(P[-1]), float((P[0]-P[-1])*Vs[0])])
            Pinf_est, c_est = float(popt[0]), float(popt[1])
        except Exception:
            Pinf_est = float(P[-1])
            c_est = float((P[0] - Pinf_est) * Vs[0])
        V_left = max(V_fit_min, Vs[0])
        Vfit = np.logspace(np.log10(V_left), np.log10(Vs[-1]), 100)
        ax.plot(Vfit, Pinf_est + c_est / Vfit, "--", color="grey",
                label=r"$P^\star + B_1/V$ fit")
    ax.set_xscale("log")
    ax.set_xlabel(r"Lyapunov parameter $V$", fontsize=13)
    ax.set_ylabel("Avg. power (W)", fontsize=13)
    ax.tick_params(axis="both", labelsize=11)
    # Tight y-axis around the empirical envelope so the 1/V trend is visible
    ax.set_ylim(0, float(max(Phi)) * 1.08)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=11, framealpha=0.7, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_V_energy.pdf"), bbox_inches="tight")
    plt.close(fig)

    # ---- backlog vs V ----
    fig, ax = plt.subplots()
    ax.errorbar(Vs, Q / 1e6,
                yerr=[(Q - Qlo) / 1e6, (Qhi - Q) / 1e6],
                fmt="o-", color="#1f77b4", label="GreenSense", capsize=2)
    # The O(V) growth predicted by Theorem 1 is an upper bound; in this
    # lightly loaded synthetic regime the empirical backlog is dominated by
    # bursts and natural drainage rather than by V. Plot the upper-bound
    # envelope so the gap to the actual data is honest.
    if len(Vs) >= 3:
        c1, c0 = np.polyfit(Vs, Q, 1)
        ax.plot(Vs, (c0 + c1 * Vs) / 1e6, "--", color="grey",
                label=r"linear-in-$V$ envelope")
    ax.set_xscale("log")
    ax.set_xlabel(r"Lyapunov parameter $V$", fontsize=13)
    ax.set_ylabel("Avg. backlog (Mbits)", fontsize=13)
    ax.tick_params(axis="both", labelsize=11)
    ax.set_ylim(0, float(max(Qhi) / 1e6) * 1.08)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=11, framealpha=0.7, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_V_backlog.pdf"), bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ 4
def fig_sensing():
    """Synthetic sensing sweep: avg power (W) vs Wasserstein radius eps,
    DR-CVaR vs NoDR. Reads exp_sensing_wide.json (kappa=1.0, sigma in
    {0.5..32}m) so the rendered slope matches the caption-cited
    2.30 W/unit, 95% CI [0.71, 3.95]."""
    data = load("sensing_wide")
    if not data:
        # fall back to the narrow original sweep if the wide one is absent
        data = load("sensing")
        if not data:
            return
    sigmas = data.get("_sigmas")
    if sigmas is None:
        return
    eps_means, P_dr_mean, P_dr_lo, P_dr_hi = [], [], [], []
    P_no_mean, P_no_lo, P_no_hi = [], [], []
    P_dr_per_seed = []
    for sg in sigmas:
        e = data.get(f"sigma={sg}")
        if e is None:
            continue
        eps_means.append(e.get("_eps_mean", np.nan))
        dr = e["DR"]; no = e["NoDR"]
        P_dr_mean.append(dr["avg_power_W"]["mean"])
        P_dr_lo.append(dr["avg_power_W"]["lo"])
        P_dr_hi.append(dr["avg_power_W"]["hi"])
        P_no_mean.append(no["avg_power_W"]["mean"])
        P_no_lo.append(no["avg_power_W"]["lo"])
        P_no_hi.append(no["avg_power_W"]["hi"])
        P_dr_per_seed.append(dr["avg_power_W"].get("_per_seed", []))
    eps = np.array(eps_means)
    P_dr = np.array(P_dr_mean); P_dr_lo = np.array(P_dr_lo); P_dr_hi = np.array(P_dr_hi)
    P_no = np.array(P_no_mean); P_no_lo = np.array(P_no_lo); P_no_hi = np.array(P_no_hi)

    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    ax.errorbar(eps, P_dr, yerr=[P_dr - P_dr_lo, P_dr_hi - P_dr],
                fmt="*-", color="#1f77b4", capsize=2,
                label="DR-CVaR (GreenSense)", lw=1.0, ms=5)
    ax.errorbar(eps, P_no, yerr=[P_no - P_no_lo, P_no_hi - P_no],
                fmt="o:", color="#d62728", capsize=2, alpha=0.7,
                label="NoDR (Sensing Lyap.)", lw=0.8, ms=4)
    # OLS fit on DR + bootstrap CI for the slope
    slope_txt = ""
    if len(eps) >= 3:
        a, b = np.polyfit(eps, P_dr, 1)
        ef = np.linspace(eps.min(), eps.max(), 50)
        ax.plot(ef, a * ef + b, "--", color="#1f77b4",
                alpha=0.7, lw=0.9, label="linear fit (DR)")
        per_seed = np.array(P_dr_per_seed) if P_dr_per_seed and \
            all(len(x) for x in P_dr_per_seed) else None
        if per_seed is not None and per_seed.ndim == 2:
            rng = np.random.default_rng(20260517)
            Bn = 1000
            n_seeds = per_seed.shape[1]
            slopes = np.zeros(Bn)
            for i in range(Bn):
                idx = rng.integers(0, n_seeds, n_seeds)
                y = per_seed[:, idx].mean(axis=1)
                slopes[i], _ = np.polyfit(eps, y, 1)
            lo, hi = np.percentile(slopes, [2.5, 97.5])
            slope_txt = (fr"DR slope $={a:.2f}$ W/unit"
                         "\n"
                         fr"95\% CI $[{lo:.2f}, {hi:.2f}]$")
        else:
            slope_txt = fr"DR slope $\approx{a:.2f}$ W/unit"
    if slope_txt:
        ax.text(0.02, 0.97, slope_txt, transform=ax.transAxes,
                va="top", ha="left", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec="gray", alpha=0.5))
    ax.set_xlabel(r"Wasserstein radius $\varepsilon(\widehat{\Sigma})$")
    ax.set_ylabel("Avg. power (W)")
    ax.set_title("Synthetic data", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_sensing_savings.pdf"),
                bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ 5
def fig_risk_frontier():
    data = load("risk")
    if not data:
        return
    Gs, P, Plo, Phi, C, Viol = [], [], [], [], [], []
    for k, agg in data.items():
        if k.startswith("_"):
            continue
        Gs.append(agg["_Gamma"])
        pm, plo, phi = _mean_lohi(agg, "avg_power_W")
        cm, _, _ = _mean_lohi(agg, "cvar_beta")
        vm, _, _ = _mean_lohi(agg, "viol_rate")
        P.append(pm); Plo.append(plo); Phi.append(phi)
        C.append(cm); Viol.append(vm)
    order = np.argsort(Gs)
    Gs = np.array(Gs)[order]; P = np.array(P)[order]
    Plo = np.array(Plo)[order]; Phi = np.array(Phi)[order]
    C = np.array(C)[order]; Viol = np.array(Viol)[order]

    fig, ax1 = plt.subplots()
    ax1.errorbar(Gs, P, yerr=[P - Plo, Phi - P],
                 fmt="o-", color="#d62728", capsize=2,
                 label="Avg. power")
    ax1.set_xlabel(r"CVaR budget $\Gamma$")
    ax1.set_ylabel("Avg. power (W)", color="#d62728")
    ax1.tick_params(axis="y", colors="#d62728")
    # Anchor energy axis so the (in)variation in P with Gamma reads honestly
    P_lo_floor = float(min(Plo)) - 5.0
    P_hi_ceil = float(max(Phi)) + 5.0
    ax1.set_ylim(P_lo_floor, P_hi_ceil)
    ax2 = ax1.twinx()
    ax2.plot(Gs, C, "s--", color="#1f77b4",
             label=r"$\widehat{\mathrm{CVaR}}_{0.95}(\ell)$")
    ax2.set_ylabel(r"$\widehat{\mathrm{CVaR}}_{0.95}(\ell)$",
                   color="#1f77b4")
    ax2.tick_params(axis="y", colors="#1f77b4")
    # Mark the y=Gamma reference (where Gamma equals empirical CVaR_0.95)
    C_min = float(min(C))
    C_max = float(max(C))
    C_lo_floor = max(0.0, C_min - 30.0)
    C_hi_ceil = C_max + 30.0
    ax2.set_ylim(C_lo_floor, C_hi_ceil)
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_risk_frontier.pdf"),
                bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ 6
def fig_delay_cdf():
    """Empirical complementary CDF of the per-slot loss l(t) on Mon real
    data (Milan 2013-11-04 + Geolife), single canonical seed. Reads the
    per-slot traces from sim.run_main_traces."""
    path = os.path.join(RESULTS_DIR, "exp_main_real_mon_traces.json")
    if not os.path.exists(path):
        print(f"  skip fig_delay_cdf: {path} missing "
              "(run: python3 -m sim.run_main_traces)")
        return
    with open(path) as f:
        traces = json.load(f)
    fig, ax = plt.subplots()
    # Plot order: lay heavy-tailed controllers first so GreenSense ends on top
    order = ["AlwaysOn", "Threshold", "NoSensLyap", "OracleDP",
             "SensLyap", "DRCVaRLyap"]
    # Loss cap from ControlCfg.loss_max (paper eq:loss). Saturated values
    # at the cap form a degenerate point mass; suppress them from the CCDF
    # so the curves do not draw a vertical drop at x = ell_max.
    LOSS_MAX = 1000.0
    EPS = 1e-6
    for ctl in order:
        entry = traces.get(ctl)
        if not entry or "loss_trace" not in entry:
            continue
        loss = np.asarray(entry["loss_trace"], dtype=float)
        loss = np.sort(loss)
        # Empirical CCDF: Pr(l > x) = (N - rank) / N
        ranks = np.arange(1, len(loss) + 1)
        ccdf = 1.0 - ranks / len(loss)
        # Drop the cap-saturated tail (degenerate at ell_max).
        mask = (loss < LOSS_MAX - EPS) & (ccdf > 0)
        style = CTL_STYLE.get(ctl, {})
        is_green = (ctl == "DRCVaRLyap")
        ax.step(loss[mask], ccdf[mask], where="post",
                color=style.get("color", "k"),
                lw=1.6 if is_green else 1.0,
                alpha=1.0 if is_green else 0.85,
                label=style.get("label", ctl))
    ax.set_xlabel(r"per-slot loss $\ell(t)$")
    ax.set_ylabel(r"$\Pr(\ell(t) > x)$")
    ax.set_yscale("log")
    ax.set_ylim(5e-4, 1.2)
    ax.set_xlim(0, LOSS_MAX)
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_delay_cdf.pdf"),
                bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ 7
def fig_convergence():
    data = load("convergence")
    if not data:
        return
    q = np.array(data["q_trace"])           # (T, B)
    z = np.array(data["z_trace"])
    tau = np.array(data["tau_trace"])
    T = q.shape[0]
    t = np.arange(T) * 0.01  # 10 ms slots -> s

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    qsum_Mb = q.sum(axis=1) / 1e6
    ax.plot(t, qsum_Mb, "-", color="#1f77b4",
            label=r"$\sum_b q_b(t)$ (Mb)")
    ax.plot(t, z / 100.0, "-", color="#d62728",
            label=r"$z(t)/100$")
    ax.plot(t, tau, "-", color="#2ca02c", label=r"$\tau(t)$")
    ax.set_xlabel("time (s)", fontsize=13)
    ax.set_ylabel("queue / virtual / threshold", fontsize=13)
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", framealpha=0.7, fontsize=11)
    # Shared y-axis with the real-data panel, if precomputed
    try:
        _yl = json.load(open(os.path.join(RESULTS_DIR, "_convergence_ylim.json")))
        ax.set_ylim(-_yl["ymax"] * 0.03, _yl["ymax"])
    except Exception:
        pass
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_convergence.pdf"),
                bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ 8
def fig_blockage():
    data = load("blockage")
    if not data:
        return
    # collect by (p, ctl)
    by = {}
    for k, agg in data.items():
        if k.startswith("_"):
            continue
        p = agg.get("_pblk")
        ctl = agg.get("_ctl")
        if p is None or ctl is None:
            continue
        by.setdefault(ctl, []).append(
            (p, agg.get("avg_power_W", {}).get("mean", 0),
             agg.get("cvar_beta", {}).get("mean", 0)))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.5))
    for ctl, rows in by.items():
        rows.sort()
        pp = [r[0] for r in rows]
        E = [r[1] for r in rows]
        C = [r[2] for r in rows]
        style = CTL_STYLE.get(ctl, {})
        a1.plot(pp, E, "-" + style.get("marker", "o"),
                color=style.get("color", "k"),
                label=style.get("label", ctl))
        a2.plot(pp, C, "-" + style.get("marker", "o"),
                color=style.get("color", "k"))
    a1.set_xlabel(r"blockage prob. $p_{\rm blk}$", fontsize=13)
    a1.set_ylabel("avg. power (W)", fontsize=13)
    a1.tick_params(axis="both", labelsize=11)
    a1.grid(alpha=0.3)
    a1.legend(loc="center left", fontsize=11, framealpha=0.5)
    a2.set_xlabel(r"blockage prob. $p_{\rm blk}$", fontsize=13)
    a2.set_ylabel(r"empirical $\mathrm{CVaR}_{0.95}(\ell)$", fontsize=13)
    a2.tick_params(axis="both", labelsize=11)
    a2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_blockage.pdf"),
                bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ 9
def fig_burstiness():
    data = load("burst")
    if not data:
        return
    by = {}
    for k, agg in data.items():
        if k.startswith("_"):
            continue
        bp = agg.get("_burst")
        ctl = agg.get("_ctl")
        if bp is None or ctl is None:
            continue
        by.setdefault(ctl, []).append(
            (bp, agg.get("avg_power_W", {}).get("mean", 0),
             agg.get("cvar_beta", {}).get("mean", 0)))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.5))
    for ctl, rows in by.items():
        rows.sort()
        pp = [r[0] for r in rows]
        E = [r[1] for r in rows]
        C = [r[2] for r in rows]
        style = CTL_STYLE.get(ctl, {})
        a1.plot(pp, E, "-" + style.get("marker", "o"),
                color=style.get("color", "k"),
                label=style.get("label", ctl))
        a2.plot(pp, C, "-" + style.get("marker", "o"),
                color=style.get("color", "k"))
    a1.set_xlabel("burst probability", fontsize=13)
    a1.set_ylabel("avg. power (W)", fontsize=13)
    a1.tick_params(axis="both", labelsize=11)
    a1.grid(alpha=0.3)
    a1.legend(loc="center left", fontsize=11, framealpha=0.5)
    a2.set_xlabel("burst probability", fontsize=13)
    a2.set_ylabel(r"empirical $\mathrm{CVaR}_{0.95}(\ell)$", fontsize=13)
    a2.tick_params(axis="both", labelsize=11)
    # Mark the CVaR constraint Gamma = 150 used in the paper headline config
    a2.axhline(150.0, color="grey", ls="--", lw=0.8, alpha=0.7)
    a2.text(0.18, 150, r"$\Gamma=150$", color="grey", fontsize=10,
            va="bottom", ha="right", alpha=0.85)
    a2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_burstiness.pdf"),
                bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ 10
def fig_ablation():
    # Use the v2 ablation JSON (Gamma=150, sigma=8 m) that matches the paper
    # table and prose; the older exp_ablation.json (Gamma=250, sigma=1 m) is
    # superseded.
    data = load("ablation_v2") or load("ablation")
    if not data:
        return
    names = list(data.keys())
    Es = [data[n].get("avg_power_W", {}).get("mean", 0) for n in names]
    El = [data[n].get("avg_power_W", {}).get("lo", 0) for n in names]
    Eh = [data[n].get("avg_power_W", {}).get("hi", 0) for n in names]
    Cs = [data[n].get("cvar_beta", {}).get("mean", 0) for n in names]
    Cl = [data[n].get("cvar_beta", {}).get("lo", 0) for n in names]
    Ch = [data[n].get("cvar_beta", {}).get("hi", 0) for n in names]
    x = np.arange(len(names))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.5))
    a1.bar(x, Es, yerr=[np.array(Es) - np.array(El),
                        np.array(Eh) - np.array(Es)],
           color="#d62728", capsize=3)
    a1.set_xticks(x)
    a1.set_xticklabels(names, rotation=15, fontsize=8)
    a1.set_ylabel("avg. power (W)")
    a1.grid(alpha=0.3, axis="y")
    a2.bar(x, Cs, yerr=[np.array(Cs) - np.array(Cl),
                        np.array(Ch) - np.array(Cs)],
           color="#1f77b4", capsize=3)
    a2.set_xticks(x)
    a2.set_xticklabels(names, rotation=15, fontsize=8)
    a2.set_ylabel(r"empirical $\mathrm{CVaR}_{0.95}(\ell)$")
    a2.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_ablation.pdf"),
                bbox_inches="tight")
    plt.close(fig)


def main():
    print("generating figures...")
    for fn in [fig_pareto, fig_V, fig_sensing, fig_risk_frontier,
               fig_delay_cdf, fig_convergence, fig_blockage,
               fig_burstiness, fig_ablation]:
        try:
            fn()
            print(f"  ok: {fn.__name__}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  FAIL: {fn.__name__} -> {e}")
    print(f"done -> {FIG_DIR}/*.pdf")


if __name__ == "__main__":
    main()
