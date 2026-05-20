"""
channel.py
----------
3GPP UMa-NLoS large-scale path loss, two-state blockage attenuation,
Rayleigh small-scale fading, and a cosine-main-lobe beamforming gain model.

Provides compute_sinr() which returns per-cell received SINR (linear) given
UE positions, blockage states, beam choices, and the cell awake mask.
"""
from __future__ import annotations
import numpy as np
from .config import SimCfg


def path_loss_dB(d_m: np.ndarray, fc_GHz: float,
                 intercept: float = 32.4, exponent: float = 31.9) -> np.ndarray:
    """3GPP UMa-NLoS-style log-distance path loss model in dB."""
    d_clipped = np.maximum(d_m, 1.0)
    return intercept + exponent * np.log10(d_clipped) + 20.0 * np.log10(fc_GHz)


def bf_gain_dB(theta_offset_rad: np.ndarray, gain_max_dB: float,
               beamwidth_rad: float = np.pi / 4) -> np.ndarray:
    """Cosine main-lobe model. Side-lobe floor at gain_max - 20 dB."""
    g_main = gain_max_dB + 10.0 * np.log10(
        np.maximum(np.cos(theta_offset_rad / beamwidth_rad) ** 2, 1e-3))
    # cap below side-lobe floor
    g_floor = gain_max_dB - 20.0
    return np.maximum(g_main, g_floor)


def compute_sinr(ue_pos: np.ndarray,
                 cell_centres: np.ndarray,
                 serving_idx: np.ndarray,
                 beams: list,
                 awake_mask: np.ndarray,
                 blockage: np.ndarray,
                 cfg: SimCfg,
                 rng: np.random.Generator) -> np.ndarray:
    """
    Vectorised per-cell SINR for the dominant UE on each serving cell.

    Parameters
    ----------
    ue_pos       : (K, 2) UE positions
    cell_centres : (B, 2) cell positions
    serving_idx  : (B,) index into ue_pos giving the served UE per cell
    beams        : list of length B of (azimuth_rad, power_W) tuples
    awake_mask   : (B,) bool array
    blockage     : (B,) bool array, True = NLoS-blocked

    Returns
    -------
    sinr : (B,) linear SINR; zero if cell asleep.
    """
    B = cell_centres.shape[0]
    sinr = np.zeros(B)

    # precompute pairwise distances (B x B) -- cell to served UE
    d_all = np.zeros((B, B))
    for b in range(B):
        ue = ue_pos[serving_idx[b]]
        d_all[:, b] = np.linalg.norm(cell_centres - ue, axis=1)

    pl_dB_all = path_loss_dB(d_all, cfg.phy.fc_GHz,
                             cfg.phy.pl_intercept_dB,
                             cfg.phy.pl_exponent)

    # blockage attenuation only on the link from cell to *its own* served UE
    block_atten_dB = np.where(blockage, cfg.phy.blockage_atten_dB, 0.0)

    for b in range(B):
        if not awake_mask[b]:
            continue
        az_b, p_b = beams[b]
        ue = ue_pos[serving_idx[b]]
        # angle from cell b to its served UE
        delta = ue - cell_centres[b]
        ang_to_ue = np.arctan2(delta[1], delta[0])
        gain_main_dB = bf_gain_dB(ang_to_ue - az_b,
                                  cfg.phy.bf_gain_max_dBi)
        rx_dBm = (10 * np.log10(p_b * 1e3) + gain_main_dB
                  - pl_dB_all[b, b] - block_atten_dB[b])

        # Rayleigh small-scale (linear)
        fading = rng.exponential(scale=1.0)
        rx_W = 10 ** ((rx_dBm - 30.0) / 10.0) * fading

        # interference from other awake cells using *side-lobe* gain
        intf_W = 0.0
        for j in range(B):
            if j == b or not awake_mask[j]:
                continue
            az_j, p_j = beams[j]
            delta_j = ue - cell_centres[j]
            ang_j = np.arctan2(delta_j[1], delta_j[0])
            gj_dB = bf_gain_dB(ang_j - az_j, cfg.phy.bf_gain_max_dBi)
            rx_int_dBm = (10 * np.log10(p_j * 1e3) + gj_dB
                          - pl_dB_all[j, b])
            intf_W += 10 ** ((rx_int_dBm - 30.0) / 10.0) * rng.exponential(1.0)

        sinr[b] = rx_W / (cfg.noise_W + intf_W + 1e-30)
    return sinr


def shannon_rate_bps(sinr_lin: np.ndarray, bw_Hz: float) -> np.ndarray:
    """Per-cell Shannon capacity in bits/second; zeros where SINR=0."""
    return bw_Hz * np.log2(1.0 + np.maximum(sinr_lin, 0.0))
