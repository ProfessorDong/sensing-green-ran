"""
metrics.py
----------
Per-run metric extraction and bootstrap confidence intervals.
"""
from __future__ import annotations
import numpy as np
from typing import Dict, Sequence


def empirical_cvar(loss: np.ndarray, beta: float) -> float:
    """Empirical CVaR_beta(loss) = mean of the worst (1-beta) fraction."""
    if len(loss) == 0:
        return 0.0
    q = np.quantile(loss, beta)
    tail = loss[loss >= q]
    if len(tail) == 0:
        return float(q)
    return float(tail.mean())


def bootstrap_ci(samples: Sequence[float], n_boot: int = 2000,
                 ci: float = 0.95,
                 rng: np.random.Generator = None) -> Dict[str, float]:
    """Bootstrap mean and CI."""
    if rng is None:
        rng = np.random.default_rng(0)
    x = np.asarray(samples, dtype=float)
    n = len(x)
    if n == 0:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0, "std": 0.0}
    if n == 1:
        return {"mean": float(x[0]), "lo": float(x[0]),
                "hi": float(x[0]), "std": 0.0}
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = x[idx].mean()
    lo, hi = np.percentile(boots, [(1 - ci) / 2 * 100,
                                   (1 + ci) / 2 * 100])
    return {"mean": float(x.mean()), "lo": float(lo),
            "hi": float(hi), "std": float(x.std(ddof=1))}


def summarize_run(power_series: np.ndarray,
                  queue_series: np.ndarray,
                  loss_series: np.ndarray,
                  delivered_bits: float,
                  z_series: np.ndarray,
                  toggles: int,
                  arrivals_series: np.ndarray,
                  beta: float,
                  Gamma: float,
                  dt_s: float) -> Dict[str, float]:
    """Compute per-run summary metrics."""
    T = len(power_series)
    energy_J_total = float(power_series.sum() * dt_s)
    avg_power_W = float(power_series.mean())
    # energy per slot (J/slot) = mean P * dt
    energy_per_slot_J = avg_power_W * dt_s
    # energy per bit (nJ / bit) -- typical scale
    e_per_bit_nJ = (energy_J_total / max(delivered_bits, 1.0)) * 1e9
    avg_backlog_bits = float(queue_series.mean())
    # Little's law -- delay ~ backlog / throughput
    avg_arrival_rate = float(arrivals_series.mean()) / dt_s  # bits/s
    if avg_arrival_rate > 0:
        avg_delay_ms = (avg_backlog_bits / avg_arrival_rate) * 1000.0
    else:
        avg_delay_ms = 0.0
    # tail delays via per-slot backlog / arrival rate (Little proxy)
    inst_delay_ms = (queue_series / max(avg_arrival_rate, 1.0)) * 1000.0
    p95_delay_ms = float(np.percentile(inst_delay_ms, 95))
    p99_delay_ms = float(np.percentile(inst_delay_ms, 99))
    cvar = empirical_cvar(loss_series, beta)
    viol_rate = float((loss_series > Gamma).mean())
    return {
        "avg_power_W": avg_power_W,
        "energy_J_total": energy_J_total,
        "energy_per_slot_J": energy_per_slot_J,
        "energy_per_bit_nJ": e_per_bit_nJ,
        "avg_backlog_bits": avg_backlog_bits,
        "avg_delay_ms": avg_delay_ms,
        "p95_delay_ms": p95_delay_ms,
        "p99_delay_ms": p99_delay_ms,
        "cvar_beta": cvar,
        "viol_rate": viol_rate,
        "toggles": int(toggles),
        "avg_z": float(z_series.mean()),
        "delivered_bits": float(delivered_bits),
    }


def aggregate(metric_list: Sequence[Dict[str, float]]) -> Dict[str, Dict]:
    """Aggregate a list of per-seed metric dicts into bootstrap-CI summary."""
    if not metric_list:
        return {}
    keys = list(metric_list[0].keys())
    rng = np.random.default_rng(0)
    out = {}
    for k in keys:
        vals = []
        for m in metric_list:
            if k not in m:
                continue
            try:
                vals.append(float(m[k]))
            except (TypeError, ValueError):
                # non-numeric (e.g. controller label); skip
                pass
        if vals:
            out[k] = bootstrap_ci(vals, rng=rng)
    return out
