"""
isac.py
-------
Bayesian ISAC pipeline: per-UE Extended Kalman Filter tracking
[px, py, vx, vy] in 2-D. The blockage probability is appended as a logistic
filter on a separate scalar.

The posterior covariance is the operational uncertainty proxy used by the
DR-CVaR controller; in particular,
    eps(Sigma_hat) = kappa * sqrt(tr(Sigma_hat)) + kappa0
maps directly to the Wasserstein radius (eq. wass-radius in the paper).
"""
from __future__ import annotations
import numpy as np
from .config import SimCfg


def epsilon_wasserstein(Sigma_hat: np.ndarray, kappa: float = 1.0,
                        kappa0: float = 0.1) -> float:
    """Compute eps = kappa * sqrt(trace(Sigma_hat)) + kappa0."""
    tr = float(np.trace(Sigma_hat))
    return kappa * np.sqrt(max(tr, 0.0)) + kappa0


class EKFTracker:
    """One EKF per UE; here we batch over all UEs for efficiency.

    State vector per UE: x = [px, py, vx, vy]
    Dynamics: constant-velocity, F = [[I, dt I], [0, I]]
    Measurement: position only, H = [I, 0], R = sigma_isac^2 I_2.
    """

    def __init__(self, cfg: SimCfg, n_ue: int, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.dt = cfg.dt_s
        self.sigma = cfg.isac.sigma_isac_m
        self.q_pos = cfg.isac.process_noise_pos_m
        self.q_vel = cfg.isac.process_noise_vel_mps
        self.n = n_ue
        # state means: (n,4); covariances: (n,4,4)
        self.x = np.zeros((n_ue, 4))
        self.P = np.tile(np.diag([10.0, 10.0, 1.0, 1.0]),
                         (n_ue, 1, 1))
        # blockage posterior probability per UE-cell (scalar Beta-like)
        self.p_blk = 0.3 * np.ones(n_ue)
        # ---- model-mismatch knobs (reviewer robustness experiment) ----
        # Per-UE bias direction (unit vector, deterministic per UE via the
        # shared rng so it is reproducible across seeds). Bias magnitude
        # is cfg.isac.bias_m. Bias is injected post-update into x[:,:2] so
        # the controller sees a shifted ISAC posterior mean.
        bias_m = float(getattr(cfg.isac, "bias_m", 0.0))
        if bias_m > 0.0:
            ang = 2.0 * np.pi * self.rng.random(n_ue)
            self._bias_vec = bias_m * np.column_stack(
                [np.cos(ang), np.sin(ang)])
        else:
            self._bias_vec = np.zeros((n_ue, 2))
        # Covariance-underestimate factor: controller's reported Sigma_hat
        # is rescaled by 1/factor^2 (over-confidence). factor=1 is the
        # calibrated baseline.
        self._cov_factor = float(getattr(cfg.isac, "sigma_truth_factor", 1.0))

    def predict(self):
        dt = self.dt
        F = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
        Q = np.diag([self.q_pos ** 2, self.q_pos ** 2,
                     self.q_vel ** 2, self.q_vel ** 2])
        self.x = (F @ self.x.T).T
        self.P = np.einsum("ij,njk,lk->nil", F, self.P, F) + Q

    def update(self, true_pos: np.ndarray):
        """Measurement update with noisy position observations."""
        z = true_pos + self.rng.normal(scale=self.sigma,
                                       size=true_pos.shape)
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        R = (self.sigma ** 2) * np.eye(2)
        # innovation
        y = z - (H @ self.x.T).T            # (n, 2)
        S = np.einsum("ij,njk,lk->nil", H, self.P, H) + R  # (n,2,2)
        # Kalman gain
        K = np.einsum("njk,lk,nlm->njm", self.P, H,
                      np.linalg.inv(S))
        # state and covariance update
        self.x = self.x + np.einsum("nij,nj->ni", K, y)
        I = np.eye(4)
        KH = np.einsum("nij,jk->nik", K, H)
        self.P = np.einsum("nij,njk->nik", I - KH, self.P)
        # ---- model-mismatch post-processing ----
        # 1) Inject constant per-UE position bias into the posterior mean
        #    (controller-observed x_hat). The true world is unaffected.
        if np.any(self._bias_vec):
            self.x[:, :2] = self.x[:, :2] + self._bias_vec
        # 2) Over-confidence: report a posterior covariance that is smaller
        #    than the actual EKF posterior by 1/factor^2. We shrink the 2x2
        #    position block of every per-UE covariance (Wasserstein radius
        #    eps = kappa*sqrt(tr(Sigma)) consumes only the trace).
        if self._cov_factor != 1.0:
            s = 1.0 / (self._cov_factor ** 2)
            self.P[:, :2, :2] = self.P[:, :2, :2] * s

    def update_blockage(self, observed: np.ndarray, alpha: float = 0.1):
        """Exponential filter on per-UE blockage probability."""
        self.p_blk = (1 - alpha) * self.p_blk + alpha * observed.astype(float)

    def trace_per_ue(self) -> np.ndarray:
        return np.trace(self.P, axis1=1, axis2=2)

    def mean_trace(self) -> float:
        return float(self.trace_per_ue().mean())
