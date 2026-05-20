"""
run_mismatch.py
---------------
Reviewer-driven model-mismatch robustness experiment.

Two sweeps, both on Mon Milan + Geolife at V=1e4, Gamma=150, beta=0.95,
T=5000 slots, 10 canonical seeds:

  Sweep A  Position bias    b in {0, 2, 5, 10} m
  Sweep B  Cov underestimate factor in {1, 2, 4, 8}
                             (sigma_reported = sigma_true / factor)

For each scenario, two controllers (SensLyap = NoDR, DRCVaRLyap = DR) are
run, aggregated over 10 seeds with bootstrap CIs, and saved to
  sim/results/exp_mismatch.json

Run:
    python -m sim.run_mismatch
"""
from __future__ import annotations
import os
import time
import json
import numpy as np

from .config import default_cfg, SimCfg
from .run import run_grid, save_json
from .metrics import aggregate


CONTROLLERS = ["SensLyap", "DRCVaRLyap"]


def _cfg_mismatch(bias_m: float = 0.0,
                  sigma_truth_factor: float = 1.0,
                  V: float = 1e4,
                  Gamma: float = 150.0,
                  beta: float = 0.95) -> SimCfg:
    """Canonical Mon Milan + Geolife setup matching the real-data table,
    with the model-mismatch knobs."""
    c = default_cfg()
    c.ctrl.V = V
    c.ctrl.Gamma = Gamma
    c.ctrl.beta = beta
    c.isac.kappa = 1.0
    c.isac.L_loss_Lipschitz = 10.0
    c.traffic_source = "milan"
    c.mobility_source = "geolife"
    c.milan_file = "milan_2013-11-04.txt"
    c.time.T_slots = 5000
    # mismatch knobs
    c.isac.bias_m = float(bias_m)
    c.isac.sigma_truth_factor = float(sigma_truth_factor)
    return c


def _agg_with_seeds(rows):
    if not rows:
        return {}
    agg = aggregate(rows)
    for k in ("avg_power_W", "cvar_beta", "viol_rate",
              "p99_delay_ms", "avg_delay_ms", "eps_wass_mean"):
        if k in agg:
            agg[k]["_per_seed"] = [float(r.get(k, 0.0)) for r in rows]
    agg["_n_seeds"] = len(rows)
    return agg


def run_all(workers: int = 16) -> dict:
    seeds = list(np.random.SeedSequence(20260517).generate_state(10))
    seeds = [int(s) for s in seeds]
    print(f"seeds = {seeds[:3]}... ({len(seeds)} total)")

    bias_values = [0.0, 2.0, 5.0, 10.0]
    factor_values = [1.0, 2.0, 4.0, 8.0]

    out = {
        "_meta": {
            "V": 1e4, "Gamma": 150.0, "beta": 0.95,
            "T_slots": 5000, "n_seeds": len(seeds),
            "traffic_source": "milan",
            "mobility_source": "geolife",
            "milan_file": "milan_2013-11-04.txt",
            "kappa": 1.0, "L_loss_Lipschitz": 10.0,
            "bias_values_m": bias_values,
            "factor_values": factor_values,
            "controllers": CONTROLLERS,
            "seeds": seeds,
        }
    }

    # ----- Sweep A: position bias -----
    t0 = time.time()
    cfgs_bias = [_cfg_mismatch(bias_m=b) for b in bias_values]
    raw_bias = run_grid(CONTROLLERS, cfgs_bias, seeds, workers=workers)
    # tag each raw row with its bias level by matching against cfg
    # (run_grid does not stamp cfg meta; we match by job order).
    # Reconstruct: rows are returned in the order (ctl, cfg, seed) iterated.
    # Easier: re-tag by iterating jobs in the same order.
    idx = 0
    jobs_order = []
    for ctl in CONTROLLERS:
        for cfg in cfgs_bias:
            for sd in seeds:
                jobs_order.append((ctl, cfg.isac.bias_m, sd))
    for r, j in zip(raw_bias, jobs_order):
        if isinstance(r, dict) and "error" not in r:
            r["_bias_m"] = j[1]
            r["_seed_idx"] = j[2]

    for b in bias_values:
        key = f"bias={b}"
        out[key] = {"_knob": "bias", "_value": b}
        for ctl in CONTROLLERS:
            rows = [r for r in raw_bias
                    if r.get("controller") == ctl
                    and "error" not in r
                    and abs(r.get("_bias_m", -1) - b) < 1e-9]
            if rows:
                out[key][ctl] = _agg_with_seeds(rows)
                P = out[key][ctl]["avg_power_W"]["mean"]
                C = out[key][ctl]["cvar_beta"]["mean"]
                V_ = out[key][ctl].get("viol_rate", {}).get("mean", 0.0)
                print(f"  bias={b:>4} {ctl:>11}: P={P:.1f}W "
                      f"CVaR={C:.1f} viol={V_:.3f}")
    print(f"  >> bias sweep done in {time.time() - t0:.1f}s")

    # ----- Sweep B: covariance underestimate -----
    t0 = time.time()
    cfgs_fac = [_cfg_mismatch(sigma_truth_factor=f) for f in factor_values]
    raw_fac = run_grid(CONTROLLERS, cfgs_fac, seeds, workers=workers)
    jobs_order = []
    for ctl in CONTROLLERS:
        for cfg in cfgs_fac:
            for sd in seeds:
                jobs_order.append((ctl, cfg.isac.sigma_truth_factor, sd))
    for r, j in zip(raw_fac, jobs_order):
        if isinstance(r, dict) and "error" not in r:
            r["_factor"] = j[1]
            r["_seed_idx"] = j[2]

    for f in factor_values:
        key = f"factor={f}"
        out[key] = {"_knob": "sigma_truth_factor", "_value": f}
        for ctl in CONTROLLERS:
            rows = [r for r in raw_fac
                    if r.get("controller") == ctl
                    and "error" not in r
                    and abs(r.get("_factor", -1) - f) < 1e-9]
            if rows:
                out[key][ctl] = _agg_with_seeds(rows)
                P = out[key][ctl]["avg_power_W"]["mean"]
                C = out[key][ctl]["cvar_beta"]["mean"]
                V_ = out[key][ctl].get("viol_rate", {}).get("mean", 0.0)
                print(f"  factor={f:>4} {ctl:>11}: P={P:.1f}W "
                      f"CVaR={C:.1f} viol={V_:.3f}")
    print(f"  >> factor sweep done in {time.time() - t0:.1f}s")

    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    cfg0 = default_cfg()
    out_path = os.path.join(cfg0.results_dir, "exp_mismatch.json")
    t0 = time.time()
    res = run_all(workers=args.workers)
    save_json(res, out_path)
    print(f"\n>> saved {out_path}   total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
