"""
run.py
------
Single-episode driver and parallel grid runner.

CLI:
    python -m sim.run all          # run every experiment + save JSONs
    python -m sim.run main         # run just the MAIN comparison
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import traceback
import multiprocessing as mp
from typing import Dict, List, Tuple
import copy
import numpy as np

from .config import SimCfg, default_cfg
from .channel import compute_sinr, shannon_rate_bps
from .mobility import MobilityGenerator, make_mobility
from .blockage import BlockageModel
from .traffic import build_arrivals, used_milan
from .isac import EKFTracker, epsilon_wasserstein
from .controllers import (CONTROLLER_REGISTRY, AlwaysOn,
                          ThresholdHeuristic, NoSensingLyapunov,
                          SensingLyapunov, DRCVaRLyapunov, OracleDP,
                          power_cell)
from .metrics import summarize_run, aggregate


# ---------------------------------------------------------------------------
def run_episode(controller_name: str, cfg: SimCfg, seed: int,
                trace: bool = False) -> Dict:
    rng = np.random.default_rng(seed)
    # ----- world components -----
    mob = make_mobility(cfg, rng)
    blk = BlockageModel(cfg, rng)
    arrivals = build_arrivals(cfg, rng)  # (B, T)
    isac = EKFTracker(cfg, mob.n_total, rng)

    # ----- initial ISAC state -----
    isac.x[:, :2] = mob.pos
    isac.x[:, 2:] = mob.vel

    # ----- controller -----
    ctrl_cls = CONTROLLER_REGISTRY[controller_name]
    controller = ctrl_cls(cfg)

    B, T = cfg.topo.B, cfg.time.T_slots
    q = np.zeros(B)
    z_series = np.zeros(T)
    power_series = np.zeros(T)
    backlog_series = np.zeros(T)
    loss_series = np.zeros(T)
    arr_series = np.zeros(T)
    if trace:
        q_trace = np.zeros((T, B))
        tau_trace = np.zeros(T)
    delivered_bits = 0.0
    toggles = 0
    s_prev = np.ones(B, dtype=int)
    # mean arrival per cell -- for loss normalization (paper's proxy)
    a_bar = arrivals.mean(axis=1)
    a_bar_safe = np.maximum(a_bar, 1.0)

    cells = cfg.topo.cell_centres()

    for t in range(T):
        # ----- world step -----
        pos, vel = mob.step()
        blk_state = blk.step()
        # ----- ISAC update -----
        isac.predict()
        isac.update(pos)
        ue_blk_obs = np.repeat(blk_state.astype(float), mob.K)
        isac.update_blockage(ue_blk_obs)
        eps_wass = epsilon_wasserstein(
            isac.P.mean(axis=0),
            cfg.isac.kappa, cfg.isac.kappa0)

        # ----- assemble state -----
        serv = mob.serving_indices()
        state = {
            "q": q.copy(),
            "x_hat": isac.x.copy(),
            "x_true": np.column_stack([pos, vel]),
            "p_blk": isac.p_blk.copy(),
            "blk_true": blk_state.copy(),
            "cells": cells,
            "serv": serv,
            "s_prev": s_prev.copy(),
            "eps_wass": eps_wass,
            "a_bar": a_bar,
            "a_true": arrivals[:, t],
        }

        # ----- controller decision -----
        try:
            action = controller.decide(state)
        except Exception as e:
            print(f"[{controller_name} seed={seed} t={t}] decide error: {e}")
            action = {"s": np.ones(B, dtype=int),
                      "beams": [(0.0, cfg.pwr.p_tx_max_W)] * B,
                      "rho": np.zeros(B)}
        s_t = action["s"]
        beams = action["beams"]

        # ----- channel & service rate -----
        sinr = compute_sinr(pos, cells, serv, beams,
                            s_t.astype(bool), blk_state, cfg, rng)
        rates_bps = shannon_rate_bps(sinr, cfg.phy.bw_Hz)
        mu = rates_bps * cfg.dt_s * s_t   # bits per slot (0 if sleeping)
        # ----- queue update -----
        served = np.minimum(q, mu)
        q = np.maximum(q - mu, 0.0) + arrivals[:, t]
        delivered_bits += float(served.sum())
        # arrivals trace = total arrived bits per slot (for Little's law)
        arr_series[t] = arrivals[:, t].sum()

        # ----- power -----
        P_t = 0.0
        for b in range(B):
            P_t += power_cell(cfg, s_t[b], beams[b][1], s_prev[b])
            if s_t[b] != s_prev[b]:
                toggles += 1
        power_series[t] = P_t

        # ----- loss & virtual-queue update (only for DRCVaR) -----
        # eq:loss: l(t) = min{ sum_b ([q_b - mu_b]_+ + a_b)/a_bar_b, ell_max }.
        # The q used here is already q(t+1) = [q(t)-mu]_+ + a(t) from the queue
        # update above, so sum(q / a_bar) equals the pre-cap loss in eq:loss.
        loss_pre = float(np.sum(q / a_bar_safe))
        loss = min(loss_pre, cfg.ctrl.loss_max)
        loss_series[t] = loss
        backlog_series[t] = float(q.sum())
        if isinstance(controller, DRCVaRLyapunov):
            controller.update_tau_z(loss, eps_wass)
            z_series[t] = controller.z
            if trace:
                tau_trace[t] = controller.tau
        else:
            z_series[t] = 0.0
            if trace:
                tau_trace[t] = 0.0

        if trace:
            q_trace[t] = q
        s_prev = s_t.copy()

    metrics = summarize_run(power_series, backlog_series, loss_series,
                            delivered_bits, z_series, toggles, arr_series,
                            cfg.ctrl.beta, cfg.ctrl.Gamma, cfg.dt_s)
    metrics["eps_wass_mean"] = float(epsilon_wasserstein(
        isac.P.mean(axis=0), cfg.isac.kappa, cfg.isac.kappa0))
    metrics["sigma_isac"] = cfg.isac.sigma_isac_m
    metrics["V"] = cfg.ctrl.V
    metrics["Gamma"] = cfg.ctrl.Gamma
    metrics["beta"] = cfg.ctrl.beta
    metrics["controller"] = controller_name

    if trace:
        return {
            "metrics": metrics,
            "q_trace": q_trace.tolist(),
            "z_trace": z_series.tolist(),
            "tau_trace": tau_trace.tolist(),
            "power_trace": power_series.tolist(),
            "loss_trace": loss_series.tolist(),
        }
    return metrics


# ---------------------------------------------------------------------------
def _run_one(args):
    name, cfg, seed = args
    try:
        return run_episode(name, cfg, seed)
    except Exception as e:
        tb = traceback.format_exc()
        return {"error": str(e), "traceback": tb,
                "controller": name, "seed": seed}


def run_grid(controllers: List[str], cfg_list: List[SimCfg],
             seeds: List[int], workers: int = 16) -> List[Dict]:
    jobs = []
    for ctl in controllers:
        for cfg in cfg_list:
            for seed in seeds:
                jobs.append((ctl, cfg, seed))
    print(f"  >> dispatching {len(jobs)} jobs over {workers} workers")
    t0 = time.time()
    with mp.get_context("spawn").Pool(processes=workers) as pool:
        out = pool.map(_run_one, jobs)
    print(f"  >> grid done in {time.time() - t0:.1f}s")
    return out


# ---------------------------------------------------------------------------
def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)


# ---------------------------------------------------------------------------
def exp_main(workers: int, seeds: List[int]) -> Dict:
    cfg = default_cfg()
    cfg.ctrl.V = 10000.0
    cfg.ctrl.Gamma = 250.0
    cfg.ctrl.beta = 0.95
    controllers = ["AlwaysOn", "Threshold", "NoSensLyap", "SensLyap",
                   "DRCVaRLyap", "OracleDP"]
    raw = run_grid(controllers, [cfg], seeds, workers=workers)
    save_json(raw, os.path.join(cfg.results_dir, "raw_main.json"))
    out = {}
    for ctl in controllers:
        rows = [r for r in raw if r.get("controller") == ctl
                and "error" not in r]
        if rows:
            out[ctl] = aggregate(rows)
            out[ctl]["_n_seeds"] = len(rows)
    return out


def exp_V_sweep(workers, seeds) -> Dict:
    # paper's V range: 10^2..10^6 (cf. Section VII Table II)
    Vs = [300.0, 1000.0, 3000.0, 10000.0, 30000.0, 100000.0, 300000.0]
    cfgs = []
    for V in Vs:
        c = default_cfg()
        c.ctrl.V = V
        c.ctrl.Gamma = 250.0
        cfgs.append(c)
    raw = run_grid(["DRCVaRLyap"], cfgs, seeds, workers=workers)
    save_json(raw, os.path.join(default_cfg().results_dir, "raw_V.json"))
    out = {}
    for V, c in zip(Vs, cfgs):
        rows = [r for r in raw if r.get("controller") == "DRCVaRLyap"
                and abs(r.get("V", 0) - V) < 1e-6 and "error" not in r]
        if rows:
            out[f"V={V}"] = aggregate(rows)
            out[f"V={V}"]["_V"] = V
    return out


def exp_sensing_sweep(workers, seeds) -> Dict:
    sigmas = [0.5, 1.0, 2.0, 4.0, 8.0]
    cfgs = []
    for sg in sigmas:
        c = default_cfg()
        c.ctrl.V = 10000.0
        c.isac.sigma_isac_m = sg
        # exaggerate kappa to make Wasserstein effect visible on energy
        c.isac.kappa = 0.5
        cfgs.append(c)
    raw = run_grid(["DRCVaRLyap"], cfgs, seeds, workers=workers)
    save_json(raw, os.path.join(default_cfg().results_dir, "raw_sensing.json"))
    out = {}
    for sg, c in zip(sigmas, cfgs):
        rows = [r for r in raw if abs(r.get("sigma_isac", 0) - sg) < 1e-6
                and "error" not in r]
        if rows:
            out[f"sigma={sg}"] = aggregate(rows)
            out[f"sigma={sg}"]["_sigma"] = sg
    return out


def exp_risk_sweep(workers, seeds) -> Dict:
    Gammas = [100.0, 150.0, 200.0, 250.0, 300.0, 400.0, 500.0]
    cfgs = []
    for G in Gammas:
        c = default_cfg()
        c.ctrl.V = 10000.0
        c.ctrl.Gamma = G
        cfgs.append(c)
    raw = run_grid(["DRCVaRLyap"], cfgs, seeds, workers=workers)
    save_json(raw, os.path.join(default_cfg().results_dir, "raw_risk.json"))
    out = {}
    for G, c in zip(Gammas, cfgs):
        rows = [r for r in raw if abs(r.get("Gamma", 0) - G) < 1e-6
                and "error" not in r]
        if rows:
            out[f"Gamma={G}"] = aggregate(rows)
            out[f"Gamma={G}"]["_Gamma"] = G
    return out


def exp_burstiness(workers, seeds) -> Dict:
    burst_probs = [0.01, 0.05, 0.10, 0.20]
    out = {}
    for bp in burst_probs:
        cfgs = []
        c = default_cfg()
        c.ctrl.V = 10000.0
        c.traf.burst_prob = bp
        cfgs.append(c)
        raw = run_grid(["AlwaysOn", "Threshold", "NoSensLyap", "SensLyap",
                        "DRCVaRLyap"], cfgs, seeds, workers=workers)
        for ctl in ["AlwaysOn", "Threshold", "NoSensLyap", "SensLyap",
                    "DRCVaRLyap"]:
            rows = [r for r in raw if r.get("controller") == ctl
                    and "error" not in r]
            if rows:
                key = f"burst={bp}|{ctl}"
                out[key] = aggregate(rows)
                out[key]["_burst"] = bp
                out[key]["_ctl"] = ctl
    return out


def exp_blockage(workers, seeds) -> Dict:
    pblk = [0.1, 0.2, 0.3, 0.5]
    out = {}
    for p in pblk:
        c = default_cfg()
        c.ctrl.V = 10000.0
        c.blk.p_block_target = p
        # adjust mean_los_s to keep mean_nlos_s and shift probability
        # stationary = p / (p + (mean_los/mean_nlos)*(1-p)*?). Simpler: rescale dwells
        # Keep mean_nlos = 1s; mean_los = (1-p)/p
        c.blk.mean_los_s = max((1 - p) / max(p, 0.01), 0.5)
        c.blk.mean_nlos_s = 1.0
        raw = run_grid(["AlwaysOn", "Threshold", "NoSensLyap", "SensLyap",
                        "DRCVaRLyap"], [c], seeds, workers=workers)
        for ctl in ["AlwaysOn", "Threshold", "NoSensLyap", "SensLyap",
                    "DRCVaRLyap"]:
            rows = [r for r in raw if r.get("controller") == ctl
                    and "error" not in r]
            if rows:
                key = f"pblk={p}|{ctl}"
                out[key] = aggregate(rows)
                out[key]["_pblk"] = p
                out[key]["_ctl"] = ctl
    return out


def exp_convergence(seeds) -> Dict:
    cfg = default_cfg()
    cfg.time.T_slots = 8000  # 80 s trace
    cfg.ctrl.V = 10000.0
    cfg.ctrl.Gamma = 400.0  # comfortable slack to show clean dynamics
    res = run_episode("DRCVaRLyap", cfg, seeds[0], trace=True)
    return res


def exp_ablation(workers, seeds) -> Dict:
    """Paper's method w/ vs w/o DR, virtual queue, sensing."""
    variants = {
        "Full":      {"use_dr": True, "use_virtq": True, "use_sensing": True},
        "NoDR":      {"use_dr": False, "use_virtq": True, "use_sensing": True},
        "NoVirtQ":   {"use_dr": True, "use_virtq": False, "use_sensing": True},
        "NoSensing": {"use_dr": True, "use_virtq": True, "use_sensing": False},
    }
    cfgs = []
    keys = []
    for name, sets in variants.items():
        c = default_cfg()
        c.ctrl.V = 10000.0  # match main experiment
        for k, v in sets.items():
            setattr(c.ctrl, k, v)
        cfgs.append(c)
        keys.append(name)
    out = {}
    raw_all = []
    for name, c in zip(keys, cfgs):
        raw = run_grid(["DRCVaRLyap"], [c], seeds, workers=workers)
        rows = [r for r in raw if "error" not in r]
        if rows:
            out[name] = aggregate(rows)
            out[name]["_variant"] = name
    save_json(raw_all, os.path.join(default_cfg().results_dir,
                                    "raw_ablation.json"))
    return out


# ---------------------------------------------------------------------------
EXPS = {
    "main":       exp_main,
    "V":          exp_V_sweep,
    "sensing":    exp_sensing_sweep,
    "risk":       exp_risk_sweep,
    "burst":      exp_burstiness,
    "blockage":   exp_blockage,
    "convergence": exp_convergence,
    "ablation":   exp_ablation,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", default="all", nargs="?",
                    choices=list(EXPS.keys()) + ["all"])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()

    seeds = list(np.random.SeedSequence(20260517).generate_state(args.seeds))
    seeds = [int(s) for s in seeds]
    print(f"seeds = {seeds[:5]}... ({len(seeds)} total)")
    cfg0 = default_cfg()
    print(f"Milan trace usable: {used_milan(cfg0)}")
    if args.which == "all":
        names = list(EXPS.keys())
    else:
        names = [args.which]

    for name in names:
        print(f"\n===== experiment: {name} =====")
        t0 = time.time()
        try:
            if name == "convergence":
                res = EXPS[name](seeds)
            else:
                res = EXPS[name](args.workers, seeds)
            save_json(res,
                      os.path.join(cfg0.results_dir, f"exp_{name}.json"))
            print(f"  >> saved exp_{name}.json   ({time.time()-t0:.1f}s)")
        except Exception as e:
            tb = traceback.format_exc()
            print(f"  >> FAILED {name}: {e}\n{tb}")


if __name__ == "__main__":
    main()
