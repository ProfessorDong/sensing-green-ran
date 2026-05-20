"""
mobility_geolife.py
-------------------
Thin facade over `sim.geolife_loader` to satisfy the documented public
interface (`from sim.mobility_geolife import load_or_build_cache`).

The actual heavy-lifting (walking `Geolife Trajectories 1.3/Data/<userID>/
Trajectory/*.plt`, equirectangular projection, ping-pong resampling, per-UE
rotation + 200 m recentering) lives in `sim/geolife_loader.py`. This module
re-exports the loader so callers can write the more semantically clear
`mobility_geolife` import path.
"""
from __future__ import annotations
from .geolife_loader import (        # noqa: F401
    load_or_build_cache,
    build_cache,
    discover_trajectories,
    select_trajectories,
    GEOLIFE_ROOT,
    DEFAULT_CACHE,
)
