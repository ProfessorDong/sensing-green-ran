"""
make_tables.py
--------------
Emit booktabs LaTeX snippets from sim/results/exp_*.json.
"""
from __future__ import annotations
import json
import os

RESULTS_DIR = "sim/results"


def load(name):
    path = os.path.join(RESULTS_DIR, f"exp_{name}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def fmt(d, key, scale=1.0, fmtstr="{:.1f}"):
    e = d.get(key, {})
    m = e.get("mean", 0.0) * scale
    lo = e.get("lo", 0.0) * scale
    hi = e.get("hi", 0.0) * scale
    halfw = max(hi - m, m - lo) / 1.0
    return fmtstr.format(m), fmtstr.format(halfw)


def table_main():
    data = load("main")
    if not data:
        return ""
    order = ["AlwaysOn", "Threshold", "NoSensLyap", "SensLyap",
             "DRCVaRLyap", "OracleDP"]
    pretty = {
        "AlwaysOn": "Always-On",
        "Threshold": "Threshold",
        "NoSensLyap": "No-Sens Lyap.",
        "SensLyap": "Sens. Lyap.",
        "DRCVaRLyap": r"\textbf{GreenSense}",
        "OracleDP": "Oracle DP (LB)",
    }
    rows = []
    for ctl in order:
        if ctl not in data:
            continue
        agg = data[ctl]
        Pm, Pw = fmt(agg, "avg_power_W")
        EBm, EBw = fmt(agg, "energy_per_bit_nJ", fmtstr="{:.1f}")
        Dm, Dw = fmt(agg, "p99_delay_ms")
        Vm, Vw = fmt(agg, "viol_rate", scale=100.0, fmtstr="{:.2f}")
        Cm, Cw = fmt(agg, "cvar_beta")
        bold = "DRCVaRLyap" == ctl
        if bold:
            rows.append(
                f"\\textbf{{GreenSense}} & $\\mathbf{{{Pm}\\!\\pm\\!{Pw}}}$ "
                f"& $\\mathbf{{{EBm}\\!\\pm\\!{EBw}}}$ "
                f"& $\\mathbf{{{Dm}\\!\\pm\\!{Dw}}}$ "
                f"& $\\mathbf{{{Vm}\\!\\pm\\!{Vw}}}$ "
                f"& $\\mathbf{{{Cm}\\!\\pm\\!{Cw}}}$ \\\\")
        else:
            rows.append(
                f"{pretty[ctl]} & ${Pm}\\!\\pm\\!{Pw}$ "
                f"& ${EBm}\\!\\pm\\!{EBw}$ "
                f"& ${Dm}\\!\\pm\\!{Dw}$ "
                f"& ${Vm}\\!\\pm\\!{Vw}$ "
                f"& ${Cm}\\!\\pm\\!{Cw}$ \\\\")
    out = (
        "\\begin{tabular}{lccccc}\n\\toprule\n"
        "Controller & Avg. power (W) & Energy/bit (nJ) "
        "& $p_{99}$ delay (ms) & CVaR violation (\\%) "
        "& $\\widehat{\\CVaR}_{0.95}(\\ell)$ \\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )
    return out


def table_sensing():
    data = load("sensing")
    if not data:
        return ""
    rows = []
    items = sorted(
        ((agg.get("_sigma", 0.0), key, agg) for key, agg in data.items()
         if not key.startswith("_")),
        key=lambda x: x[0])
    for sg, _, agg in items:
        Em, _, _ = (agg.get("eps_wass_mean", {}).get("mean", 0), 0, 0)
        Pm, Pw = fmt(agg, "avg_power_W")
        Cm, Cw = fmt(agg, "cvar_beta")
        Vm, Vw = fmt(agg, "viol_rate", scale=100.0, fmtstr="{:.2f}")
        rows.append(
            f"{sg:.1f} & {Em:.3f} & ${Pm}\\!\\pm\\!{Pw}$ "
            f"& ${Cm}\\!\\pm\\!{Cw}$ "
            f"& ${Vm}\\!\\pm\\!{Vw}$ \\\\")
    out = (
        "\\begin{tabular}{ccccc}\n\\toprule\n"
        "$\\sigma_{\\rm ISAC}$ (m) & $\\bar\\varepsilon(\\widehat\\Sigma)$ "
        "& Avg. power (W) "
        "& $\\widehat{\\CVaR}_{0.95}(\\ell)$ "
        "& CVaR violation (\\%) \\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )
    return out


def table_ablation():
    data = load("ablation")
    if not data:
        return ""
    pretty = {"Full": "Full (GreenSense)",
              "NoDR": r"$\setminus$\,DR ($\varepsilon{=}0$)",
              "NoVirtQ": r"$\setminus$\,virtual queue",
              "NoSensing": r"$\setminus$\,ISAC mean"}
    rows = []
    for k in ["Full", "NoDR", "NoVirtQ", "NoSensing"]:
        if k not in data:
            continue
        agg = data[k]
        Pm, Pw = fmt(agg, "avg_power_W")
        Cm, Cw = fmt(agg, "cvar_beta")
        Vm, Vw = fmt(agg, "viol_rate", scale=100.0, fmtstr="{:.2f}")
        Dm, Dw = fmt(agg, "p99_delay_ms")
        name = pretty.get(k, k)
        rows.append(
            f"{name} & ${Pm}\\!\\pm\\!{Pw}$ "
            f"& ${Dm}\\!\\pm\\!{Dw}$ "
            f"& ${Cm}\\!\\pm\\!{Cw}$ "
            f"& ${Vm}\\!\\pm\\!{Vw}$ \\\\")
    out = (
        "\\begin{tabular}{lcccc}\n\\toprule\n"
        "Variant & Avg. power (W) & $p_{99}$ delay (ms) "
        "& $\\widehat{\\CVaR}_{0.95}(\\ell)$ "
        "& violation (\\%) \\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )
    return out


def main():
    out_dir = RESULTS_DIR
    os.makedirs(out_dir, exist_ok=True)
    for name, fn in [("main", table_main),
                     ("sensing", table_sensing),
                     ("ablation", table_ablation)]:
        txt = fn()
        if txt:
            path = os.path.join(out_dir, f"table_{name}.tex")
            with open(path, "w") as f:
                f.write(txt)
            print(f"  wrote {path}")
        else:
            print(f"  skipping {name}: no data")


if __name__ == "__main__":
    main()
