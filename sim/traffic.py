"""
traffic.py
----------
Per-cell arrival process a_b(t) in *bits per slot*.

Primary source: Milan Telecom 2013 cellular dataset (file
sim/data/milan_2013-11-04.txt), tab-separated columns:
    <cellID(1..10000)> <interval_ms> <countryCode>
    <SMS_in> <SMS_out> <Call_in> <Call_out> <Internet>
Blanks (empty fields between tabs) are treated as 0. We aggregate over all
country codes per (cellID, timestamp) into a single Internet-activity
scalar (the dataset's natural per-cell-per-10min observable), then take
a *contiguous 7-cell block* from the centre of Milan's 100x100 grid (cells
indexed 1..10000 row-major: row = (cell-1)//100, col = (cell-1)%100; centre
cells live around the 5050..5056 strip). The block is interpolated from
the 144 ten-minute bins to the simulator's T-slot grid; we pick the 50 s
window starting at the cell-busy hour (21:00 local). Per-cell rates are
then linearly rescaled so each cell's time-average matches
cfg.traf.base_rate_mbps. Pareto bursts (existing synthetic model) are
applied on top.

Fallback: sinusoidal diurnal mean + per-slot Pareto bursts. Used iff the
Milan file is absent or unparseable.
"""
from __future__ import annotations
import os
import numpy as np
from typing import Optional, Tuple, Dict
from .config import SimCfg


MILAN_GRID = 100        # Milan grid is 100x100 (cellIDs 1..10000)
MILAN_BINS_PER_DAY = 144  # 10-min bins
MILAN_BIN_MS = 600_000

# Process-wide LRU cache of parsed Milan diurnal (so multiple build_arrivals
# calls within a single process don't reparse the 350 MB file).
_MILAN_CACHE: dict = {}


def _parse_float(s: str) -> float:
    s = s.strip()
    if not s:
        return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def _try_load_milan(path: str, B: int,
                    centre_cell: int = 5050,
                    verbose: bool = False
                    ) -> Optional[Tuple[np.ndarray, list, np.ndarray]]:
    """Parse Milan trace -> (B, 144) Internet activity, list of cell IDs used,
    and the (144,) timestamp grid in ms. Returns None on failure.

    Strategy:
      * one pass over the file, tab-split, fold over country codes;
      * accumulate sum(internet) into a (10000, 144) dense matrix;
      * pick a contiguous 7-cell horizontal block centred on the most
        active cell within the central 10x10 patch (rows 49-51, cols 47-53);
      * return that block.
    """
    if not os.path.exists(path) or os.path.getsize(path) < 1024:
        return None
    cache_key = (path, B, centre_cell)
    if cache_key in _MILAN_CACHE:
        return _MILAN_CACHE[cache_key]
    # Also try an on-disk numpy cache (the parsed (B,144) block) so subprocess
    # workers don't repeat the 3-4 s parse. Key by file basename so that
    # different Milan days (e.g. Mon 2013-11-04 vs Fri 2013-11-15) don't share
    # a cache.
    base = os.path.splitext(os.path.basename(path))[0]
    disk_cache = os.path.join(os.path.dirname(path),
                              f"_milan_block_B{B}_{base}.npz")
    if os.path.exists(disk_cache) and os.path.getmtime(disk_cache) >= os.path.getmtime(path):
        try:
            d = np.load(disk_cache, allow_pickle=True)
            res = (d["block"], list(d["cell_ids"]), d["ts_grid"])
            _MILAN_CACHE[cache_key] = res
            return res
        except Exception:
            pass
    try:
        ncells = MILAN_GRID * MILAN_GRID
        nbins = MILAN_BINS_PER_DAY
        activity = np.zeros((ncells, nbins), dtype=np.float64)
        t0_ms: Optional[int] = None

        with open(path, "r", buffering=1 << 20) as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3:
                    continue
                try:
                    cid = int(parts[0])
                    t_ms = int(parts[1])
                except ValueError:
                    continue
                if cid < 1 or cid > ncells:
                    continue
                if t0_ms is None:
                    t0_ms = t_ms
                bin_idx = (t_ms - t0_ms) // MILAN_BIN_MS
                if bin_idx < 0 or bin_idx >= nbins:
                    continue
                # Internet is column index 7 (0-based) when present.
                inet = _parse_float(parts[7]) if len(parts) >= 8 else 0.0
                activity[cid - 1, bin_idx] += inet
        if t0_ms is None:
            return None
        ts_grid = t0_ms + np.arange(nbins) * MILAN_BIN_MS

        # ----- pick a contiguous 7-cell block in central Milan -----
        # cell (r,c) -> index r*100 + c (0-based). Centre patch r in [49,51],
        # c in [47, 53] gives the 7-wide horizontal block.
        # Pick the row (49,50,51) with maximum total activity, then take cells
        # row*100 + (47..53).
        best_row = -1
        best_total = -1.0
        for r in [49, 50, 51]:
            idxs = [r * MILAN_GRID + c for c in range(47, 54)]
            tot = float(activity[idxs].sum())
            if tot > best_total:
                best_total = tot
                best_row = r
        block_idx = [best_row * MILAN_GRID + c for c in range(47, 54)]
        block = activity[block_idx]
        cell_ids = [i + 1 for i in block_idx]  # back to 1-based IDs
        if verbose:
            print(f"[milan] picked row={best_row}, "
                  f"cell IDs (1-based)={cell_ids}, "
                  f"total Internet activity={best_total:.1f}")
        if block.shape != (B, nbins):
            # if B != 7, fall back to top-B by variance (rare; we always B=7)
            var_per_cell = activity.var(axis=1)
            top_idx = np.argsort(var_per_cell)[-B:][::-1]
            block = activity[top_idx]
            cell_ids = [int(i) + 1 for i in top_idx]
        res = (block, cell_ids, ts_grid)
        _MILAN_CACHE[cache_key] = res
        try:
            np.savez_compressed(disk_cache,
                                block=block,
                                cell_ids=np.array(cell_ids),
                                ts_grid=ts_grid)
        except Exception:
            pass
        return res
    except Exception as e:
        if verbose:
            print(f"[milan] parse failed: {e}")
        return None


def _pick_window_indices(activity_diurnal: np.ndarray,
                         T_slots: int, dt_s: float,
                         cell_busy_hour_local: int = 21,
                         tz_offset_hours_from_first_bin: int = 1
                         ) -> Tuple[float, float]:
    """Return (t_window_start_h, t_window_end_h) in hours into the day of
    the 50 s window we will simulate. We anchor to 21:00 local time of the
    file's first sample. activity_diurnal is shape (B, 144).
    """
    # The file's first bin is 00:00 CET in 2013-11-04. So bin 6*hour_local
    # corresponds to hour_local. We pick the 50 s window starting at that
    # bin. (T_slots * dt_s = 50 s by default.)
    start_h = float(cell_busy_hour_local)
    end_h = start_h + (T_slots * dt_s) / 3600.0
    return start_h, end_h


def build_arrivals(cfg: SimCfg, rng: np.random.Generator,
                   return_meta: bool = False
                   ):
    """Build a (B, T) array of bits-per-slot arrivals.

    If `return_meta=True`, also returns a dict with the Milan diurnal pattern
    used (per-cell rates over 24 h in Mbit/s, plus the simulation window).
    Used by the figure pipeline for `fig_milan_diurnal`.
    """
    B, T = cfg.topo.B, cfg.time.T_slots
    dt = cfg.dt_s
    base_bps = cfg.traf.base_rate_mbps * 1e6
    base_bits_per_slot = base_bps * dt

    use_milan = getattr(cfg, "traffic_source", "auto") in ("milan", "auto")
    milan_meta: Dict = {}

    milan = None
    if use_milan:
        milan_fname = getattr(cfg, "milan_file", "milan_2013-11-04.txt")
        milan_path = os.path.join(cfg.data_dir, milan_fname)
        milan = _try_load_milan(milan_path, B)

    if milan is not None:
        block, cell_ids, ts_grid = milan
        # ----- window pick -----
        # 144 bins span 24 h. 21:00 local = bin index 21*6 = 126. We want
        # the busiest *contiguous* 30-min from 21:00..21:30 (3 bins), but
        # since T*dt = 50 s << 600 s, we simply interpolate the 3-bin block
        # to T slots so the simulation actually rides the very busy hour.
        cell_busy_hour = 21
        bin_start = cell_busy_hour * 6  # = 126
        bin_end = bin_start + 3         # 3 bins = 30 min
        block_bsh = block[:, bin_start:bin_end + 1]  # (B, 4) endpoints

        # ----- interpolate to T slots -----
        # Convert each cell's Internet "activity" units to bits-per-slot,
        # rescaled so its TIME-AVERAGE OVER THE WHOLE DAY equals
        # base_bits_per_slot.
        day_mean = block.mean(axis=1, keepdims=True) + 1e-9
        scale = base_bits_per_slot / day_mean         # (B, 1)
        block_scaled_day = block * scale              # (B, 144) bits/slot
        # window-restricted curve at bin_start..bin_end
        window_curve = block_scaled_day[:, bin_start:bin_end + 1]  # (B, 4)
        # x axis: bin offsets 0..3 in 10-min units -> seconds 0..1800
        x_src = np.linspace(0.0, (bin_end - bin_start) * 600.0,
                            window_curve.shape[1])  # seconds
        x_tgt = np.arange(T) * dt                    # seconds, 0..T*dt
        # if T*dt > 1800 we'd extrapolate; clamp
        x_tgt = np.minimum(x_tgt, x_src[-1])
        rates = np.zeros((B, T))
        for b in range(B):
            rates[b] = np.interp(x_tgt, x_src, window_curve[b])
        # Poisson noise about the mean rate
        a = rng.poisson(np.maximum(rates, 1.0)).astype(np.float64)

        milan_meta = {
            "cell_ids": cell_ids,
            "ts_grid_ms": ts_grid.tolist(),
            "block_diurnal_bits_per_slot": block_scaled_day,  # (B,144)
            "window_bin_start": int(bin_start),
            "window_bin_end": int(bin_end),
            "cell_busy_hour": int(cell_busy_hour),
            "mean_rate_mbps_per_cell": (
                (rates.mean(axis=1) / dt) / 1e6).tolist(),
            "peak_rate_mbps_per_cell": (
                (rates.max(axis=1) / dt) / 1e6).tolist(),
            "source": "milan",
            "milan_file": getattr(cfg, "milan_file",
                                  "milan_2013-11-04.txt"),
        }
        if return_meta:
            # one-time diagnostic line so callers can see exactly which trace
            # and cells were used per the user request
            print(f"Milan {milan_meta['milan_file']}: "
                  f"cells={cell_ids}, T_bins=144, "
                  f"mean Mbit/s/cell="
                  f"{[round(x, 2) for x in milan_meta['mean_rate_mbps_per_cell']]}")
    else:
        # ----- sinusoidal fallback -----
        t_axis = np.arange(T)
        diurnal = 1.0 + cfg.traf.diurnal_amp * np.sin(
            2 * np.pi * t_axis / T * 1.0 - np.pi / 2)
        rates = base_bits_per_slot * diurnal[None, :] * np.ones((B, 1))
        offsets = rng.uniform(0.8, 1.2, size=B)[:, None]
        rates = rates * offsets
        a = rng.poisson(rates).astype(np.float64)
        milan_meta = {"source": "synthetic"}

    # Heavy-tailed bursts: keep regardless of source.
    burst_mask = rng.random((B, T)) < cfg.traf.burst_prob
    burst_mag = (rng.pareto(cfg.traf.pareto_shape, (B, T)) + 1.0) \
        * cfg.traf.pareto_loc_scale
    a = np.where(burst_mask, a * burst_mag, a)

    if return_meta:
        return a, milan_meta
    return a


def used_milan(cfg: SimCfg) -> bool:
    fname = getattr(cfg, "milan_file", "milan_2013-11-04.txt")
    path = os.path.join(cfg.data_dir, fname)
    return os.path.exists(path) and os.path.getsize(path) > 1024
