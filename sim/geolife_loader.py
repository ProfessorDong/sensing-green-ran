"""
geolife_loader.py
-----------------
Microsoft Geolife GPS trajectories loader for the sensing-green-RAN simulator.

Pipeline
========
1. Walk `sim/data/geolife/Geolife Trajectories 1.3/Data/<userID>/Trajectory/*.plt`.
2. Parse the 6-header-line .plt format (lat, lon, alt, alt_ft, excel_date, date, time).
3. Keep trajectories that
   (a) have >= 1000 samples,
   (b) lie inside the Beijing urban-dense bounding box
       (lat in [39.85, 40.05], lon in [116.25, 116.50]).
4. Project (lat, lon) onto a local Cartesian frame in metres via an
   equirectangular projection centred on the bbox centre.
5. Resample each trajectory to a uniform `dt_ms` grid (default 10 ms) by
   linear interpolation. If the trajectory is shorter than the simulator
   horizon `T*dt`, loop it (ping-pong) so it tiles the horizon.
6. Per-UE recentre: translate so the start point falls inside a 200 m
   disc around the network centre. Apply a random rotation per UE to
   decorrelate the 28 UEs.
7. Cache `pos[ue, t, 2]` and `vel[ue, t, 2]` to a single .npz file.

The cache is rebuilt iff missing or if `force_rebuild=True`.

Typical usage:
    from sim.geolife_loader import load_or_build_cache
    pos, vel = load_or_build_cache(cfg)   # pos: (n_ue, T, 2) in metres
"""
from __future__ import annotations
import os
import sys
import time
import glob
import math
import numpy as np
from typing import List, Tuple, Optional

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
GEOLIFE_ROOT = os.path.join(
    "sim", "data", "geolife", "Geolife Trajectories 1.3", "Data")
DEFAULT_CACHE = os.path.join("sim", "data", "geolife_cache.npz")

# Beijing dense-urban bounding box
BJ_LAT_MIN, BJ_LAT_MAX = 39.85, 40.05
BJ_LON_MIN, BJ_LON_MAX = 116.25, 116.50

# Required minimum samples per trajectory after filtering
MIN_SAMPLES = 1000

# Recentering disc radius around the network origin (metres)
RECENTRE_RADIUS_M = 200.0

# Earth radius for equirectangular projection
R_EARTH_M = 6_371_000.0


# ---------------------------------------------------------------------------
# Geo helpers
# ---------------------------------------------------------------------------
def _bbox_centre() -> Tuple[float, float]:
    return (0.5 * (BJ_LAT_MIN + BJ_LAT_MAX),
            0.5 * (BJ_LON_MIN + BJ_LON_MAX))


def _equirectangular_xy(lat: np.ndarray, lon: np.ndarray,
                        lat0: float, lon0: float) -> Tuple[np.ndarray, np.ndarray]:
    """Equirectangular projection (lat, lon) -> (x, y) metres, around (lat0, lon0)."""
    lat_r = np.deg2rad(lat)
    lon_r = np.deg2rad(lon)
    lat0_r = math.radians(lat0)
    lon0_r = math.radians(lon0)
    x = (lon_r - lon0_r) * math.cos(lat0_r) * R_EARTH_M
    y = (lat_r - lat0_r) * R_EARTH_M
    return x, y


# ---------------------------------------------------------------------------
# .plt parser
# ---------------------------------------------------------------------------
def _parse_plt(path: str) -> Optional[np.ndarray]:
    """Return (n, 3) array of [lat, lon, t_seconds_since_first_sample] or None."""
    try:
        # Skip 6 header lines; columns: lat, lon, alt, alt_ft, excel_date, date, time
        # Excel date column 4 (0-indexed) is the most reliable monotonic clock.
        arr = np.loadtxt(path, delimiter=",", skiprows=6, usecols=(0, 1, 4))
    except (ValueError, OSError, IndexError):
        return None
    if arr.ndim != 2 or arr.shape[0] < MIN_SAMPLES:
        return None
    # excel_date is in days since 1899-12-30; convert to seconds and zero out.
    t_sec = (arr[:, 2] - arr[0, 2]) * 86400.0
    # Drop non-monotone (rare GPS glitches) by taking strictly-increasing prefix
    keep = np.concatenate(([True], np.diff(t_sec) > 0))
    if keep.sum() < MIN_SAMPLES:
        return None
    return np.column_stack([arr[keep, 0], arr[keep, 1], t_sec[keep]])


def _inside_bbox(latlon: np.ndarray) -> bool:
    """Return True iff >= 80% of samples fall inside the Beijing bbox."""
    lat = latlon[:, 0]
    lon = latlon[:, 1]
    inside = ((lat >= BJ_LAT_MIN) & (lat <= BJ_LAT_MAX)
              & (lon >= BJ_LON_MIN) & (lon <= BJ_LON_MAX))
    return float(inside.mean()) >= 0.8


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------
def _resample_uniform(t_src: np.ndarray, xy_src: np.ndarray,
                      t_grid: np.ndarray) -> np.ndarray:
    """Linear interpolation of (x,y) onto a uniform time grid.

    If t_grid extends beyond t_src, use ping-pong tiling so the trajectory
    keeps moving with realistic kinematics (rather than freezing at the end).
    """
    T_src = t_src[-1] - t_src[0]
    if T_src <= 0:
        return np.tile(xy_src[0:1], (len(t_grid), 1))
    # Build extended source by reflecting the trajectory (ping-pong) until
    # it covers t_grid[-1].
    cycles_needed = int(np.ceil((t_grid[-1] - t_grid[0] + 1.0) / T_src)) + 1
    if cycles_needed > 1:
        t_ext = [t_src - t_src[0]]
        xy_ext = [xy_src]
        for c in range(1, cycles_needed):
            if c % 2 == 1:                       # reverse
                t_ext.append(t_ext[-1][-1] + (t_src[-1] - t_src[::-1]))
                xy_ext.append(xy_src[::-1])
            else:
                t_ext.append(t_ext[-1][-1] + (t_src - t_src[0]))
                xy_ext.append(xy_src)
        t_src2 = np.concatenate(t_ext)
        xy_src2 = np.vstack(xy_ext)
        # ensure strictly increasing (small numerical bumps possible at joins)
        keep = np.concatenate(([True], np.diff(t_src2) > 0))
        t_src2 = t_src2[keep]
        xy_src2 = xy_src2[keep]
    else:
        t_src2 = t_src - t_src[0]
        xy_src2 = xy_src
    x_g = np.interp(t_grid - t_grid[0], t_src2, xy_src2[:, 0])
    y_g = np.interp(t_grid - t_grid[0], t_src2, xy_src2[:, 1])
    return np.column_stack([x_g, y_g])


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def discover_trajectories(root: str = GEOLIFE_ROOT,
                          max_files: Optional[int] = None) -> List[str]:
    """Yield .plt paths under `root`."""
    pattern = os.path.join(root, "*", "Trajectory", "*.plt")
    paths = sorted(glob.glob(pattern))
    if max_files is not None:
        paths = paths[:max_files]
    return paths


def select_trajectories(n_ue: int,
                        root: str = GEOLIFE_ROOT,
                        verbose: bool = False) -> List[Tuple[str, np.ndarray]]:
    """Walk the dataset, pick the first `n_ue` trajectories that pass filters.

    Returns a list of (path, latlon_t_array) tuples.
    """
    paths = discover_trajectories(root)
    if verbose:
        print(f"[geolife] scanning {len(paths)} .plt files...")
    selected = []
    n_seen = 0
    for p in paths:
        n_seen += 1
        rec = _parse_plt(p)
        if rec is None:
            continue
        if not _inside_bbox(rec[:, :2]):
            continue
        selected.append((p, rec))
        if verbose and len(selected) % 10 == 0:
            print(f"[geolife]   selected {len(selected)}/{n_ue} "
                  f"(seen {n_seen})")
        if len(selected) >= n_ue:
            break
    if verbose:
        print(f"[geolife] final: {len(selected)} trajectories "
              f"selected from {n_seen} scanned.")
    return selected


def build_cache(n_ue: int, dt_s: float, T_slots: int,
                seed: int = 20260517,
                root: str = GEOLIFE_ROOT,
                cache_path: str = DEFAULT_CACHE,
                verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Build the (n_ue, T_slots, 2) position and velocity tensors and cache."""
    t0 = time.time()
    selected = select_trajectories(n_ue, root=root, verbose=verbose)
    if len(selected) < n_ue:
        raise RuntimeError(
            f"Only {len(selected)} Geolife trajectories pass filters; "
            f"need {n_ue}.")
    lat0, lon0 = _bbox_centre()
    rng = np.random.default_rng(seed)
    T = T_slots
    pos = np.zeros((n_ue, T, 2), dtype=np.float32)
    vel = np.zeros((n_ue, T, 2), dtype=np.float32)
    paths_used: List[str] = []
    n_samples_total = 0
    for u, (path, rec) in enumerate(selected):
        lat = rec[:, 0]
        lon = rec[:, 1]
        t_s = rec[:, 2]
        n_samples_total += len(lat)
        x, y = _equirectangular_xy(lat, lon, lat0, lon0)
        xy_src = np.column_stack([x, y])
        t_grid = np.arange(T, dtype=np.float64) * dt_s
        xy = _resample_uniform(t_s, xy_src, t_grid)
        # ----- random rotation per UE -----
        theta = rng.uniform(0.0, 2 * np.pi)
        c, s = math.cos(theta), math.sin(theta)
        R = np.array([[c, -s], [s, c]])
        xy = xy @ R.T
        # ----- recentre so start lands inside 200 m disc -----
        r_off = RECENTRE_RADIUS_M * math.sqrt(rng.random())
        ang_off = rng.uniform(0.0, 2 * np.pi)
        start_target = np.array([r_off * math.cos(ang_off),
                                 r_off * math.sin(ang_off)])
        shift = start_target - xy[0]
        xy = xy + shift
        # ----- finite-difference velocity -----
        v = np.zeros_like(xy)
        v[1:] = (xy[1:] - xy[:-1]) / dt_s
        v[0] = v[1]
        # ----- cap insane velocities (rare GPS jumps) -----
        v_norm = np.linalg.norm(v, axis=1)
        cap = 30.0  # m/s ~ 108 km/h; anything above is a GPS glitch
        bad = v_norm > cap
        if bad.any():
            v[bad] = v[bad] * (cap / np.maximum(v_norm[bad, None], 1e-9))
        pos[u] = xy.astype(np.float32)
        vel[u] = v.astype(np.float32)
        paths_used.append(path)

    meta = {
        "n_ue": n_ue,
        "T_slots": T,
        "dt_s": dt_s,
        "n_samples_total": int(n_samples_total),
        "bbox": [BJ_LAT_MIN, BJ_LAT_MAX, BJ_LON_MIN, BJ_LON_MAX],
        "bbox_centre": [lat0, lon0],
        "seed": seed,
        "build_seconds": float(time.time() - t0),
    }
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(
        cache_path,
        pos=pos, vel=vel,
        paths=np.array(paths_used, dtype=object),
        meta_keys=np.array(list(meta.keys()), dtype=object),
        meta_vals=np.array([str(v) for v in meta.values()], dtype=object),
    )
    if verbose:
        size_mb = os.path.getsize(cache_path) / 1e6
        print(f"[geolife] cache written -> {cache_path} "
              f"({size_mb:.2f} MB, build={meta['build_seconds']:.1f}s)")
    return pos, vel, meta


def load_or_build_cache(cfg=None,
                        n_ue: Optional[int] = None,
                        dt_s: Optional[float] = None,
                        T_slots: Optional[int] = None,
                        cache_path: str = DEFAULT_CACHE,
                        force_rebuild: bool = False,
                        verbose: bool = True
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """Load pos[ue,t,2] and vel[ue,t,2] from cache, building it if needed.

    If `cfg` is supplied (SimCfg), we derive n_ue, dt_s, T_slots from it
    unless explicitly overridden.
    """
    if cfg is not None:
        if n_ue is None:
            n_ue = cfg.topo.B * cfg.mob.n_ue_per_cell
        if dt_s is None:
            dt_s = cfg.dt_s
        if T_slots is None:
            T_slots = cfg.time.T_slots
    if n_ue is None or dt_s is None or T_slots is None:
        raise ValueError(
            "Either cfg or all of (n_ue, dt_s, T_slots) must be provided.")
    if (not force_rebuild) and os.path.exists(cache_path):
        try:
            with np.load(cache_path, allow_pickle=True) as d:
                pos = d["pos"]
                vel = d["vel"]
            if pos.shape[0] >= n_ue and pos.shape[1] >= T_slots:
                # slice exactly to what's requested
                if verbose:
                    print(f"[geolife] cache hit: {cache_path} "
                          f"(have {pos.shape}, want ({n_ue},{T_slots},2))")
                return pos[:n_ue, :T_slots, :], vel[:n_ue, :T_slots, :]
            else:
                if verbose:
                    print(f"[geolife] cache too small "
                          f"(have {pos.shape}, want ({n_ue},{T_slots},2));"
                          f" rebuilding.")
        except Exception as e:
            if verbose:
                print(f"[geolife] cache load failed ({e}); rebuilding.")
    pos, vel, _ = build_cache(n_ue, dt_s, T_slots,
                              cache_path=cache_path, verbose=verbose)
    return pos, vel


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    import argparse
    ap = argparse.ArgumentParser(description="Build Geolife cache.")
    ap.add_argument("--n_ue", type=int, default=28)
    ap.add_argument("--dt_ms", type=float, default=10.0)
    ap.add_argument("--T_slots", type=int, default=8000)
    ap.add_argument("--cache", type=str, default=DEFAULT_CACHE)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    pos, vel, meta = build_cache(args.n_ue,
                                 dt_s=args.dt_ms * 1e-3,
                                 T_slots=args.T_slots,
                                 cache_path=args.cache)
    print("meta:", meta)
    print(f"pos shape: {pos.shape}, vel shape: {vel.shape}")
    print(f"pos range x: [{pos[..., 0].min():.1f}, {pos[..., 0].max():.1f}] m")
    print(f"pos range y: [{pos[..., 1].min():.1f}, {pos[..., 1].max():.1f}] m")
    print(f"|v| mean: {np.linalg.norm(vel, axis=-1).mean():.2f} m/s, "
          f"max: {np.linalg.norm(vel, axis=-1).max():.2f} m/s")


if __name__ == "__main__":
    main()
