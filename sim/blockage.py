"""
blockage.py
-----------
Two-state (LoS, NLoS-blocked) Markov chain per UE-cell link, calibrated to
3GPP TR 38.901 Block-A: stationary blockage probability ~0.3, mean LoS dwell
5 s, mean NLoS dwell 1 s.
"""
from __future__ import annotations
import numpy as np
from .config import SimCfg


class BlockageModel:
    """One state per (UE, cell) link; for tractability we model one indicator
    per cell that is shared by all UEs served at that cell at the slot."""

    def __init__(self, cfg: SimCfg, rng: np.random.Generator):
        self.rng = rng
        self.B = cfg.topo.B
        dt = cfg.dt_s
        # transition probabilities per slot (geometric dwell)
        self.p_los_to_nlos = dt / cfg.blk.mean_los_s
        self.p_nlos_to_los = dt / cfg.blk.mean_nlos_s
        # initial draw from stationary
        p_stat = cfg.blk.p_block_target
        self.state = (rng.random(self.B) < p_stat).astype(np.int8)  # 1 = blocked

    def step(self) -> np.ndarray:
        u = self.rng.random(self.B)
        new = self.state.copy()
        # LoS (0) -> NLoS (1)
        mask_los = (self.state == 0)
        new[mask_los & (u < self.p_los_to_nlos)] = 1
        # NLoS (1) -> LoS (0)
        mask_nlos = (self.state == 1)
        new[mask_nlos & (u < self.p_nlos_to_los)] = 0
        self.state = new
        return self.state.astype(bool)

    def get(self) -> np.ndarray:
        return self.state.astype(bool)
