"""
mobility.py
-----------
Bounded random-waypoint UE mobility per cell. Returns UE positions at every
slot.

Two concrete generators with the same .step() / .serving_indices() interface:

* `MobilityGenerator` - synthetic bounded random-waypoint (default).
* `GeolifeReplayMobility` - replays cached Microsoft Geolife trajectories
  (see `sim.mobility_geolife.load_or_build_cache`). Pre-projected, per-UE
  rotated, and recentred to a 200 m disc around the network origin.

Mobility here just produces trajectories used by the channel and ISAC
modules; arrival-rate diurnal modulation lives in traffic.py.
"""
from __future__ import annotations
import os
import numpy as np
from .config import SimCfg


class MobilityGenerator:
    """Generates a sequence of UE positions and velocities."""

    def __init__(self, cfg: SimCfg, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.B = cfg.topo.B
        self.K = cfg.mob.n_ue_per_cell  # per-cell UE count
        self.cell_centres = cfg.topo.cell_centres()
        # global UE list; each cell owns K UEs (sequential indexing)
        self.n_total = self.B * self.K
        # initial positions inside each cell's radius
        r = cfg.mob.waypoint_radius_m * np.sqrt(rng.random(self.n_total))
        ang = 2 * np.pi * rng.random(self.n_total)
        base = np.repeat(self.cell_centres, self.K, axis=0)
        self.pos = base + np.column_stack(
            [r * np.cos(ang), r * np.sin(ang)])
        # waypoints
        self.target = self._draw_waypoint()
        # speeds
        self.speed = rng.uniform(cfg.mob.speed_min_mps,
                                 cfg.mob.speed_max_mps, size=self.n_total)
        # velocity from speed toward target
        self.vel = self._compute_vel()

    def _draw_waypoint(self) -> np.ndarray:
        r = self.cfg.mob.waypoint_radius_m * np.sqrt(self.rng.random(self.n_total))
        ang = 2 * np.pi * self.rng.random(self.n_total)
        base = np.repeat(self.cell_centres, self.K, axis=0)
        return base + np.column_stack([r * np.cos(ang), r * np.sin(ang)])

    def _compute_vel(self) -> np.ndarray:
        delta = self.target - self.pos
        d = np.linalg.norm(delta, axis=1, keepdims=True) + 1e-9
        return self.speed[:, None] * delta / d

    def step(self) -> tuple:
        """Advance one slot. Returns (positions (n_total,2), velocities)."""
        dt = self.cfg.dt_s
        self.pos = self.pos + self.vel * dt
        # if reached waypoint -> new waypoint
        d_target = np.linalg.norm(self.target - self.pos, axis=1)
        reach = d_target < 1.0
        if np.any(reach):
            new_wp = self._draw_waypoint()
            self.target[reach] = new_wp[reach]
            self.speed[reach] = self.rng.uniform(self.cfg.mob.speed_min_mps,
                                                 self.cfg.mob.speed_max_mps,
                                                 size=int(reach.sum()))
            self.vel = self._compute_vel()
        return self.pos.copy(), self.vel.copy()

    def serving_indices(self) -> np.ndarray:
        """Pick one UE per cell with the strongest geometry (nearest)."""
        idx = np.zeros(self.B, dtype=int)
        for b in range(self.B):
            ues = np.arange(b * self.K, (b + 1) * self.K)
            d = np.linalg.norm(self.pos[ues] - self.cell_centres[b], axis=1)
            idx[b] = ues[int(np.argmin(d))]
        return idx


# ---------------------------------------------------------------------------
# Geolife replay
# ---------------------------------------------------------------------------
class GeolifeReplayMobility:
    """Replay cached Geolife trajectories with the same API as
    `MobilityGenerator`. The cache (pos[n_ue,T,2], vel[n_ue,T,2]) is built
    by `sim.geolife_loader.load_or_build_cache` on first use and reused
    thereafter.

    Notes
    -----
    * `n_ue = B * K` UEs are sliced from the cache (the first B*K = 28
      trajectories for the default B=7, K=4 layout). Cache is consistent
      across seeds; per-seed randomness comes from the channel / blockage /
      arrival processes, not from the trajectory ordering.
    * `serving_indices()` returns the per-cell nearest UE (same convention
      as the synthetic generator).
    """

    def __init__(self, cfg: SimCfg, rng: np.random.Generator):
        # Local import to avoid pulling matplotlib/Geolife when the simulator
        # is asked to use the synthetic path only.
        from .geolife_loader import load_or_build_cache

        self.cfg = cfg
        self.rng = rng
        self.B = cfg.topo.B
        self.K = cfg.mob.n_ue_per_cell
        self.n_total = self.B * self.K
        self.cell_centres = cfg.topo.cell_centres()

        pos_full, vel_full = load_or_build_cache(
            cfg=cfg, n_ue=self.n_total,
            dt_s=cfg.dt_s, T_slots=cfg.time.T_slots, verbose=False)
        # cache may be bigger than what we need; slice exactly
        self.pos_traj = np.asarray(
            pos_full[:self.n_total, :cfg.time.T_slots, :], dtype=np.float64)
        self.vel_traj = np.asarray(
            vel_full[:self.n_total, :cfg.time.T_slots, :], dtype=np.float64)
        # Re-bind each UE to a *cell* by offsetting its starting position so
        # the UE begins inside its cell's radius. This preserves the natural
        # Geolife motion (shape + speed) while keeping per-cell associations
        # similar to the synthetic generator.
        offsets = np.zeros((self.n_total, 2))
        r = cfg.mob.waypoint_radius_m * np.sqrt(rng.random(self.n_total))
        ang = 2 * np.pi * rng.random(self.n_total)
        cell_of = np.repeat(self.cell_centres, self.K, axis=0)
        wanted_start = cell_of + np.column_stack(
            [r * np.cos(ang), r * np.sin(ang)])
        offsets = wanted_start - self.pos_traj[:, 0, :]
        # apply offset to whole trajectory
        self.pos_traj = self.pos_traj + offsets[:, None, :]
        # velocities unchanged

        self.t = 0
        self.pos = self.pos_traj[:, 0, :].copy()
        self.vel = self.vel_traj[:, 0, :].copy()

    def step(self) -> tuple:
        T = self.cfg.time.T_slots
        # advance with replay (clamp at the last sample if T_slots in cache
        # is less than wanted; load_or_build_cache normally guarantees >=T)
        self.t = min(self.t + 1, T - 1)
        self.pos = self.pos_traj[:, self.t, :].copy()
        self.vel = self.vel_traj[:, self.t, :].copy()
        return self.pos.copy(), self.vel.copy()

    def serving_indices(self) -> np.ndarray:
        idx = np.zeros(self.B, dtype=int)
        for b in range(self.B):
            ues = np.arange(b * self.K, (b + 1) * self.K)
            d = np.linalg.norm(self.pos[ues] - self.cell_centres[b], axis=1)
            idx[b] = ues[int(np.argmin(d))]
        return idx


def make_mobility(cfg: SimCfg, rng: np.random.Generator):
    """Factory dispatch on cfg.mobility_source in {synthetic, geolife}."""
    src = getattr(cfg, "mobility_source", "synthetic")
    if src == "geolife":
        return GeolifeReplayMobility(cfg, rng)
    return MobilityGenerator(cfg, rng)
