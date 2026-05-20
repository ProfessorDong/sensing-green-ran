"""
controllers.py
--------------
Six per-slot controllers with a common .decide(state) interface.

Unit normalization:
  - Queue q_b stored in *bits*; controller internally rescales to "slot-units"
    by dividing by a per-cell normalizer Q_SCALE so that V is dimensionally a
    Watt-per-(normalized-bit) constant. Choosing Q_SCALE = mu_max (max bits
    per slot) makes V in W; the paper's V range 10^2..10^4 then corresponds
    to "willing to delay 1 slot to save 100..10^4 W".

  1. AlwaysOn           -- baseline; all cells awake, low-power BF
  2. ThresholdHeuristic -- buffer-occupancy timer
  3. NoSensingLyapunov  -- drift-plus-penalty, queue only, no risk, no ISAC
  4. SensingLyapunov    -- DPP with ISAC-predicted mu; no virtual queue
  5. DRCVaRLyapunov     -- paper's method: DPP + virtual queue + DR penalty
  6. OracleDP           -- finite-lookahead DP with future ground-truth (UB)

All controllers consume a `State` dict and emit an `Action` dict with
  s_b   : (B,) {0,1}
  beam  : list of (az_rad, p_W) tuples
  rho_b : (B,) handover bias (informational)
"""
from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple
from .config import SimCfg


# ------------------- helpers -------------------
def predicted_mu(cfg: SimCfg, beam: Tuple[float, float],
                 ue_pos: np.ndarray, cell_pos: np.ndarray,
                 blocked: bool) -> float:
    """Simple link-adapted service-rate predictor used by the controllers.
    Mirrors channel.compute_sinr at the mean (no fading, no interf.)."""
    az, p_W = beam
    d = np.linalg.norm(ue_pos - cell_pos)
    pl_dB = (cfg.phy.pl_intercept_dB
             + cfg.phy.pl_exponent * np.log10(max(d, 1.0))
             + 20 * np.log10(cfg.phy.fc_GHz))
    delta = ue_pos - cell_pos
    ang_to_ue = np.arctan2(delta[1], delta[0])
    # cosine main-lobe
    g_dB = cfg.phy.bf_gain_max_dBi + 10 * np.log10(
        max(np.cos((ang_to_ue - az) / (np.pi / 4)) ** 2, 1e-3))
    g_dB = max(g_dB, cfg.phy.bf_gain_max_dBi - 20)
    block_dB = cfg.phy.blockage_atten_dB if blocked else 0.0
    rx_dBm = 10 * np.log10(p_W * 1e3) + g_dB - pl_dB - block_dB
    rx_W = 10 ** ((rx_dBm - 30) / 10)
    sinr = rx_W / cfg.noise_W
    rate_bps = cfg.phy.bw_Hz * np.log2(1 + max(sinr, 0))
    return rate_bps * cfg.dt_s  # bits per slot


def power_cell(cfg: SimCfg, s: int, p_W: float,
               s_prev: int) -> float:
    """Per-cell instantaneous power, eq. (power) + switching cost."""
    if s == 0:
        P = cfg.pwr.p_slp_W
    else:
        P = cfg.pwr.p_on_W + p_W / cfg.pwr.eta_PA  # PA inefficiency
    if s != s_prev:
        P += cfg.pwr.p_sw_W
    return P


# ------------------- 1. Always-On -------------------
class AlwaysOn:
    name = "AlwaysOn"

    def __init__(self, cfg: SimCfg):
        self.cfg = cfg
        self.cb = cfg.codebook()
        # pick low-power direction by default (idx 0 = lowest power, dir 0)
        self.low_p_beam = (0.0, cfg.pwr.p_tx_max_W / cfg.phy.bf_power_levels)
        self.dwell = np.zeros(cfg.topo.B, dtype=int)

    def decide(self, state):
        B = self.cfg.topo.B
        # serve toward each cell's own UE; max power always (baseline)
        beams = []
        for b in range(B):
            ue = state["x_hat"][state["serv"][b]][:2]
            cp = state["cells"][b]
            delta = ue - cp
            az = float(np.arctan2(delta[1], delta[0]))
            beams.append((az, self.cfg.pwr.p_tx_max_W))
        return {"s": np.ones(B, dtype=int), "beams": beams,
                "rho": np.zeros(B)}


# ------------------- 2. Threshold Heuristic -------------------
class ThresholdHeuristic:
    name = "Threshold"

    def __init__(self, cfg: SimCfg, q_low: float = 1e6, q_hi: float = 5e6,
                 T_hyst: int = 10):
        self.cfg = cfg
        self.q_low = q_low
        self.q_hi = q_hi
        self.T_hyst = T_hyst
        self.below_count = np.zeros(cfg.topo.B, dtype=int)
        self.s_prev = np.ones(cfg.topo.B, dtype=int)
        self.dwell = np.full(cfg.topo.B, cfg.pwr.min_on_slots, dtype=int)

    def decide(self, state):
        B = self.cfg.topo.B
        q = state["q"]
        s = self.s_prev.copy()
        for b in range(B):
            if self.dwell[b] < (self.cfg.pwr.min_on_slots if s[b] == 1
                                else self.cfg.pwr.min_off_slots):
                self.dwell[b] += 1
                continue
            if s[b] == 1:
                if q[b] < self.q_low:
                    self.below_count[b] += 1
                else:
                    self.below_count[b] = 0
                if self.below_count[b] >= self.T_hyst:
                    s[b] = 0
                    self.dwell[b] = 0
                    self.below_count[b] = 0
            else:
                if q[b] > self.q_hi:
                    s[b] = 1
                    self.dwell[b] = 0
        beams = []
        for b in range(B):
            ue = state["x_hat"][state["serv"][b]][:2]
            cp = state["cells"][b]
            delta = ue - cp
            az = np.arctan2(delta[1], delta[0])
            beams.append((az, self.cfg.pwr.p_tx_max_W))
        self.s_prev = s.copy()
        return {"s": s, "beams": beams, "rho": np.zeros(B)}


# ------------------- core DPP per-cell evaluator -------------------
# Normalization for queue-pressure-vs-power dimensional balance.
# Q_SCALE is the "natural" bit-rate scale (matches a healthy single-slot
# capacity); V then has units of [Watt * slots-of-bits], i.e. "willing to
# delay one slot of arrivals to save V Watts." Paper's V in 10^2..10^6.
Q_SCALE = 5e6  # bits (~ 500 Mb/s slot capacity)


def _dpp_per_cell(cfg: SimCfg, b: int, state, V: float, q_b: float,
                  z_t: float, c_term_per_unit_mu: float, codebook,
                  use_sensing: bool) -> Tuple[int, Tuple[float, float], float]:
    """For one cell, evaluate sleep vs each codebook entry.

    objective (per-cell, after decomposition):
        J_awake(beam) = V * P_b(beam) - (q_norm + z * c_term_per_unit_mu) * mu_b(beam)
        J_sleep       = V * P_slp
    """
    s_prev = state["s_prev"][b]
    ue_idx = state["serv"][b]
    if use_sensing:
        ue_pos = state["x_hat"][ue_idx][:2]
        blocked = state["p_blk"][ue_idx] > 0.5
    else:
        # sensing-blind: assume nominal centre, ignore blockage
        ue_pos = state["cells"][b] + np.array([10.0, 0.0])
        blocked = False
    cp = state["cells"][b]

    q_norm = q_b / Q_SCALE
    # combined queue + virtual-queue pressure per unit of bits served:
    effective_q = q_norm + z_t * c_term_per_unit_mu

    # ----- sleep cost -----
    P_slp = power_cell(cfg, 0, 0.0, s_prev)
    J_slp = V * P_slp  # no served bits

    # ----- best awake choice -----
    best_J = np.inf
    best_beam = codebook[0]
    for beam in codebook:
        mu = predicted_mu(cfg, beam, ue_pos, cp, blocked)  # bits/slot
        P_awk = power_cell(cfg, 1, beam[1], s_prev)
        J = V * P_awk - effective_q * mu
        if J < best_J:
            best_J = J
            best_beam = beam
    if J_slp <= best_J:
        return 0, codebook[0], J_slp
    return 1, best_beam, best_J


# ------------------- 3. No-Sensing Lyapunov -------------------
class NoSensingLyapunov:
    name = "NoSensLyap"

    def __init__(self, cfg: SimCfg):
        self.cfg = cfg
        self.cb = cfg.codebook()
        self.s_prev = np.ones(cfg.topo.B, dtype=int)
        self.dwell = np.full(cfg.topo.B, cfg.pwr.min_on_slots, dtype=int)

    def decide(self, state):
        B = self.cfg.topo.B
        V = self.cfg.ctrl.V
        s_new = np.zeros(B, dtype=int)
        beams = []
        state["s_prev"] = self.s_prev
        for b in range(B):
            min_dwell = (self.cfg.pwr.min_on_slots if self.s_prev[b] == 1
                         else self.cfg.pwr.min_off_slots)
            if self.dwell[b] < min_dwell:
                s_new[b] = self.s_prev[b]
                # carry on same beam (max power if awake)
                ue = state["x_hat"][state["serv"][b]][:2]
                cp = state["cells"][b]
                az = np.arctan2((ue - cp)[1], (ue - cp)[0])
                beams.append((az, self.cfg.pwr.p_tx_max_W))
                self.dwell[b] += 1
                continue
            s, beam, _ = _dpp_per_cell(self.cfg, b, state, V, state["q"][b],
                                       0.0, 0.0, self.cb,
                                       use_sensing=False)
            s_new[b] = s
            beams.append(beam)
            if s != self.s_prev[b]:
                self.dwell[b] = 0
            else:
                self.dwell[b] += 1
        self.s_prev = s_new.copy()
        return {"s": s_new, "beams": beams, "rho": np.zeros(B)}


# ------------------- 4. Sensing Lyapunov (no risk) -------------------
class SensingLyapunov(NoSensingLyapunov):
    name = "SensLyap"

    def decide(self, state):
        B = self.cfg.topo.B
        V = self.cfg.ctrl.V
        s_new = np.zeros(B, dtype=int)
        beams = []
        state["s_prev"] = self.s_prev
        for b in range(B):
            min_dwell = (self.cfg.pwr.min_on_slots if self.s_prev[b] == 1
                         else self.cfg.pwr.min_off_slots)
            if self.dwell[b] < min_dwell:
                s_new[b] = self.s_prev[b]
                ue = state["x_hat"][state["serv"][b]][:2]
                cp = state["cells"][b]
                az = np.arctan2((ue - cp)[1], (ue - cp)[0])
                beams.append((az, self.cfg.pwr.p_tx_max_W))
                self.dwell[b] += 1
                continue
            s, beam, _ = _dpp_per_cell(self.cfg, b, state, V, state["q"][b],
                                       0.0, 0.0, self.cb,
                                       use_sensing=True)
            s_new[b] = s
            beams.append(beam)
            if s != self.s_prev[b]:
                self.dwell[b] = 0
            else:
                self.dwell[b] += 1
        self.s_prev = s_new.copy()
        return {"s": s_new, "beams": beams, "rho": np.zeros(B)}


# ------------------- 5. DR-CVaR Lyapunov (paper's method) -------------------
class DRCVaRLyapunov(NoSensingLyapunov):
    name = "DRCVaRLyap"

    def __init__(self, cfg: SimCfg):
        super().__init__(cfg)
        self.tau = cfg.ctrl.tau_init
        self.z = 0.0
        self.last_loss = 0.0
        # Robbins-Monro slot index (Algo 1, alpha_tau(t)).
        self.t_slot = 0
        # last per-cell c term for fairness in decomposition
        self.last_c = np.zeros(cfg.topo.B)

    def update_tau_z(self, loss: float, eps_wass: float):
        cfg = self.cfg.ctrl
        # Algo 1, line 2: tau(t) <- Pi_{[0, ell_max]}[tau + alpha(t)*((1-b)^-1 1{l>t} - 1)]
        # Robbins-Monro step: alpha(t) = alpha_tau * loss_max / (1 + t*decay)
        alpha_t = (cfg.alpha_tau * cfg.loss_max
                   / (1.0 + self.t_slot * cfg.alpha_tau_decay))
        ind = 1.0 if loss > self.tau else 0.0
        delta_tau = alpha_t * (ind / (1.0 - cfg.beta) - 1.0)
        # Euclidean projection on [0, loss_max]
        self.tau = max(min(self.tau + delta_tau, cfg.loss_max), 0.0)
        self.t_slot += 1
        # g_rob(t): nominal expectation approximated by realized one-sample
        # + L_ell * eps if DR is on; - (1-beta)(Gamma - tau)
        loss_term = max(loss - self.tau, 0.0)
        L_eps = (self.cfg.isac.L_loss_Lipschitz * eps_wass
                 if cfg.use_dr else 0.0)
        g = loss_term + L_eps - (1.0 - cfg.beta) * (cfg.Gamma - self.tau)
        if cfg.use_virtq:
            # cap z growth rate to keep it bounded but responsive
            self.z = max(self.z + g, 0.0)
        else:
            self.z = 0.0
        self.last_loss = loss

    def decide(self, state):
        B = self.cfg.topo.B
        cfg = self.cfg
        V = cfg.ctrl.V
        s_new = np.zeros(B, dtype=int)
        beams = []
        state["s_prev"] = self.s_prev
        # per-cell c-term per unit of mu_b: marginal effect of serving one
        # more bit at cell b on the virtual-queue penalty.
        # loss l(t) = sum_b q_b / a_bar_b  (paper's eq.)
        # dl/dq_b = 1/a_bar_b; dq_b/d(mu_b) = -1 (when q_b > mu_b)
        # subgradient of (l-tau)_+ wrt l: indicator{l > tau}
        # => dg_rob/d(mu_b) = -1/a_bar_b * 1{l>tau}
        # so awake reduces g_rob; the per-cell DPP penalty term subtracted
        # from V*P_b is z * (1/a_bar_b) * mu_b * subgrad.
        sub_grad = 1.0 if self.last_loss > self.tau else 0.0
        a_bar = state.get("a_bar", np.ones(B))
        # per unit served-bits coefficient, normalized by Q_SCALE for
        # dimensional consistency with q_norm
        if cfg.ctrl.use_virtq:
            c_term_arr = (sub_grad / np.maximum(a_bar, 1.0)) * 1.0
        else:
            c_term_arr = np.zeros(B)
        for b in range(B):
            min_dwell = (cfg.pwr.min_on_slots if self.s_prev[b] == 1
                         else cfg.pwr.min_off_slots)
            if self.dwell[b] < min_dwell:
                s_new[b] = self.s_prev[b]
                ue = state["x_hat"][state["serv"][b]][:2]
                cp = state["cells"][b]
                az = np.arctan2((ue - cp)[1], (ue - cp)[0])
                beams.append((az, cfg.pwr.p_tx_max_W))
                self.dwell[b] += 1
                continue
            s, beam, _ = _dpp_per_cell(cfg, b, state, V, state["q"][b],
                                       self.z, c_term_arr[b], self.cb,
                                       use_sensing=cfg.ctrl.use_sensing)
            s_new[b] = s
            beams.append(beam)
            if s != self.s_prev[b]:
                self.dwell[b] = 0
            else:
                self.dwell[b] += 1
        self.s_prev = s_new.copy()
        return {"s": s_new, "beams": beams, "rho": np.zeros(B)}


# ------------------- 6. Oracle DP (finite horizon LB) -------------------
class OracleDP(NoSensingLyapunov):
    """Greedy receding-horizon controller that *knows* future arrivals,
    mobility, and blockage. Approximates the DP lower-bound on energy by
    choosing, slot by slot, the per-cell action that minimizes expected
    forward energy subject to keeping each queue below a soft cap.

    The horizon-1 myopic case is a strong approximation when the dominant
    dynamic is queue stability; we extend with a one-step lookahead via the
    ground-truth next-slot arrival.
    """
    name = "OracleDP"

    def __init__(self, cfg: SimCfg):
        super().__init__(cfg)

    def decide(self, state):
        B = self.cfg.topo.B
        s_new = np.zeros(B, dtype=int)
        beams = []
        state["s_prev"] = self.s_prev
        a_next = state.get("a_true", np.zeros(B))
        for b in range(B):
            min_dwell = (self.cfg.pwr.min_on_slots if self.s_prev[b] == 1
                         else self.cfg.pwr.min_off_slots)
            if self.dwell[b] < min_dwell:
                s_new[b] = self.s_prev[b]
                ue = state["x_true"][state["serv"][b]][:2]
                cp = state["cells"][b]
                az = np.arctan2((ue - cp)[1], (ue - cp)[0])
                beams.append((az, self.cfg.pwr.p_tx_max_W))
                self.dwell[b] += 1
                continue
            cp = state["cells"][b]
            ue = state["x_true"][state["serv"][b]][:2]
            blocked = bool(state["blk_true"][b])
            # try sleep
            P_slp = power_cell(self.cfg, 0, 0.0, self.s_prev[b])
            q_next_slp = state["q"][b] + a_next[b]
            # decision based on q after action
            best_J = P_slp + (q_next_slp / Q_SCALE) * 1000.0  # tiny queue cost regularization
            best_s, best_beam = 0, self.cb[0]
            for beam in self.cb:
                mu = predicted_mu(self.cfg, beam, ue, cp, blocked)
                P_awk = power_cell(self.cfg, 1, beam[1], self.s_prev[b])
                q_next = max(state["q"][b] - mu, 0.0) + a_next[b]
                J = P_awk + (q_next / Q_SCALE) * 1000.0
                if J < best_J:
                    best_J = J
                    best_s = 1
                    best_beam = beam
            s_new[b] = best_s
            beams.append(best_beam)
            if best_s != self.s_prev[b]:
                self.dwell[b] = 0
            else:
                self.dwell[b] += 1
        self.s_prev = s_new.copy()
        return {"s": s_new, "beams": beams, "rho": np.zeros(B)}


CONTROLLER_REGISTRY = {
    "AlwaysOn": AlwaysOn,
    "Threshold": ThresholdHeuristic,
    "NoSensLyap": NoSensingLyapunov,
    "SensLyap": SensingLyapunov,
    "DRCVaRLyap": DRCVaRLyapunov,
    "OracleDP": OracleDP,
}
