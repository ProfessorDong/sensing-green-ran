"""
config.py
---------
Centralized, calibrated simulation parameters for the sensing-aware green RAN
controller experiments. All numerical defaults are chosen to be representative
of an urban-micro 5G mmWave + sub-6 hybrid deployment with realistic 3GPP-
style path loss, EARTH-derived power, and an ISAC-driven Bayesian tracker.

Parameters here are the single source of truth for every other module.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple, List
import numpy as np


# ---------- Topology ----------
@dataclass
class TopologyCfg:
    B: int = 7                       # number of cells (centre + 6 ring)
    isd_m: float = 200.0             # inter-site distance (urban micro)

    def cell_centres(self) -> np.ndarray:
        """Hex layout: centre at origin, six cells at radius isd_m."""
        centres = [(0.0, 0.0)]
        for k in range(6):
            ang = np.pi / 3.0 * k
            centres.append((self.isd_m * np.cos(ang),
                            self.isd_m * np.sin(ang)))
        return np.array(centres[: self.B])


# ---------- Time ----------
@dataclass
class TimeCfg:
    dt_ms: float = 10.0              # slot duration: 10 ms
    T_slots: int = 5000              # 50 s per run


# ---------- PHY / channel ----------
@dataclass
class PhyCfg:
    fc_GHz: float = 28.0             # mmWave carrier
    bw_Hz: float = 100e6             # per-cell bandwidth
    p_tx_max_dBm: float = 30.0       # 1 W transmit antenna ref; cap at 30 dBm
    noise_dBm: float = -84.0         # thermal noise across 100 MHz @ NF=10 dB
    bf_gain_max_dBi: float = 18.0    # peak array gain
    bf_codebook_dirs: int = 8        # 8 azimuth directions
    bf_power_levels: int = 3         # 3 PA-level rungs => 24-entry codebook
    pl_intercept_dB: float = 32.4    # 3GPP UMa NLoS
    pl_exponent: float = 31.9        # ditto
    blockage_atten_dB: float = 25.0  # mmWave LoS blockage drop
    rician_K_dB: float = 3.0         # small-scale fading proxy


# ---------- Energy (EARTH-style) ----------
@dataclass
class PowerCfg:
    p_on_W: float = 130.0            # active baseline (RF chain + BB)
    p_slp_W: float = 8.0             # deep sleep power
    p_tx_max_W: float = 20.0         # PA cap at full BF
    eta_PA: float = 0.30             # PA efficiency
    p_sw_W: float = 5.0              # one-toggle switching energy proxy
    min_on_slots: int = 5            # min dwell awake after wake
    min_off_slots: int = 5           # min dwell asleep after sleep


# ---------- Traffic ----------
@dataclass
class TrafficCfg:
    base_rate_mbps: float = 30.0     # mean per-cell rate during day
    diurnal_amp: float = 0.45        # 1 +/- diurnal swing
    pareto_shape: float = 1.5        # heavy-tail burst
    pareto_loc_scale: float = 3.0    # scale factor on bursts (multiplier)
    burst_prob: float = 0.05         # fraction of slots with a burst
    seed_traffic: int = 0


# ---------- Blockage ----------
@dataclass
class BlockageCfg:
    p_block_target: float = 0.30     # stationary blockage probability
    mean_los_s: float = 5.0          # mean LoS dwell (s)
    mean_nlos_s: float = 1.0         # mean blocked dwell (s)


# ---------- Mobility ----------
@dataclass
class MobilityCfg:
    n_ue_per_cell: int = 4
    speed_min_mps: float = 1.0
    speed_max_mps: float = 3.0
    waypoint_radius_m: float = 100.0  # bounded random walk per cell


# ---------- ISAC ----------
@dataclass
class IsacCfg:
    sigma_isac_m: float = 1.0        # measurement std (position) -- swept
    process_noise_pos_m: float = 0.05
    process_noise_vel_mps: float = 0.10
    kappa: float = 0.10              # Wasserstein radius slope
    kappa0: float = 0.05             # residual radius (bias absorber)
    L_loss_Lipschitz: float = 1.0    # L_ell in eq. (grob_dual)
    # ---- model-mismatch knobs (reviewer-driven robustness experiment) ----
    # Constant per-UE position bias (m) injected into the ISAC posterior
    # mean x_hat (random unit-direction per UE, seeded). Controller does not
    # know about the bias. Default 0.0 = nominal (no mismatch).
    bias_m: float = 0.0
    # Controller's reported sigma_meas vs true. true sigma is the EKF's
    # actual measurement noise (sigma_isac_m); the controller's Sigma_hat
    # is rescaled by 1/factor^2 to simulate an over-confident filter that
    # reports a smaller posterior covariance than is justified. factor=1
    # is calibrated; factor>1 is over-confident.
    sigma_truth_factor: float = 1.0


# ---------- Controller ----------
@dataclass
class ControlCfg:
    V: float = 10000.0               # Lyapunov tradeoff knob (sweepable)
    beta: float = 0.95               # CVaR confidence
    Gamma: float = 250.0             # CVaR budget on loss l(t)
    # Loss cap ell_max (paper eq:loss). l(t) <= ell_max; tau in [0, ell_max].
    # 1000.0 sits comfortably above all observed CVaR_beta values so the cap
    # is a feasibility-of-tracking envelope rather than a binding constraint.
    loss_max: float = 1000.0
    # Robbins-Monro step: alpha_tau(t) = alpha_tau * loss_max / (1 + t*decay).
    # Default tuned so alpha_tau(0) approx matches the prior constant step
    # 0.002 * Gamma = 0.3 used in earlier rounds (alpha_tau(0)=0.3 at
    # loss_max=1000).
    alpha_tau: float = 3e-4          # tau learning rate (Algo 1, line 2)
    alpha_tau_decay: float = 1e-3    # R-M decay: step ~ 1/(1+t*decay)
    tau_init: float = 0.0
    use_dr: bool = True              # ablation toggle: DR term on/off
    use_virtq: bool = True           # ablation toggle: virtual queue on/off
    use_sensing: bool = True         # ablation toggle: ISAC mean on/off
    hysteresis_slots: int = 5        # min sleep/awake dwell


# ---------- Top-level bundle ----------
@dataclass
class SimCfg:
    topo: TopologyCfg = field(default_factory=TopologyCfg)
    time: TimeCfg = field(default_factory=TimeCfg)
    phy: PhyCfg = field(default_factory=PhyCfg)
    pwr: PowerCfg = field(default_factory=PowerCfg)
    traf: TrafficCfg = field(default_factory=TrafficCfg)
    blk: BlockageCfg = field(default_factory=BlockageCfg)
    mob: MobilityCfg = field(default_factory=MobilityCfg)
    isac: IsacCfg = field(default_factory=IsacCfg)
    ctrl: ControlCfg = field(default_factory=ControlCfg)
    data_dir: str = "sim/data"
    results_dir: str = "sim/results"
    # ---- data-source knobs (added in round-3 real-data validation) ----
    # traffic_source: 'auto' (= prefer Milan if file exists, else synthetic),
    #                 'milan' (force Milan; falls back to synthetic on parse
    #                   failure), 'synthetic' (skip Milan entirely).
    traffic_source: str = "auto"
    # mobility_source: 'synthetic' (bounded RWP) or 'geolife' (cached real
    # GPS trajectories from Microsoft Geolife, recentred + rotated).
    mobility_source: str = "synthetic"
    # milan_file: filename (relative to data_dir) of the Milan trace to use
    # when traffic_source != 'synthetic'. Defaults to Monday 2013-11-04.
    # Switch to "milan_2013-11-15.txt" for Friday.
    milan_file: str = "milan_2013-11-04.txt"

    # ----- helpers -----
    @property
    def dt_s(self) -> float:
        return self.time.dt_ms * 1e-3

    @property
    def noise_W(self) -> float:
        return 10.0 ** ((self.phy.noise_dBm - 30.0) / 10.0)

    @property
    def p_tx_max_lin_W(self) -> float:
        return self.pwr.p_tx_max_W

    def codebook(self) -> List[Tuple[float, float]]:
        """Return list of (azimuth_rad, power_W) pairs."""
        dirs = np.linspace(0.0, 2 * np.pi, self.phy.bf_codebook_dirs,
                           endpoint=False)
        powers = np.linspace(self.pwr.p_tx_max_W / self.phy.bf_power_levels,
                             self.pwr.p_tx_max_W,
                             self.phy.bf_power_levels)
        return [(float(a), float(p)) for a in dirs for p in powers]


def default_cfg(**overrides) -> SimCfg:
    cfg = SimCfg()
    for k, v in overrides.items():
        # support nested override via dotted keys ('ctrl.V': 100)
        if "." in k:
            head, tail = k.split(".", 1)
            sub = getattr(cfg, head)
            setattr(sub, tail, v)
        else:
            setattr(cfg, k, v)
    return cfg
