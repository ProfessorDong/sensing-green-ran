"""
run_main_traces.py
------------------
Dump per-slot loss traces for the six controllers under the Mon real-data
configuration (Milan 2013-11-04 + Geolife), single canonical seed.

Used by sim.make_figures::fig_delay_cdf to render a true empirical CCDF of
the per-slot QoS loss l(t), replacing the four-point summary sketch.

Run (from project root):
    python3 -m sim.run_main_traces
"""
from __future__ import annotations
import json
import os
import time
import numpy as np

from .real_data_validation import _cfg_real, CONTROLLERS_MAIN
from .run import run_episode

OUT_PATH = "sim/results/exp_main_real_mon_traces.json"


def main():
    seeds = list(np.random.SeedSequence(20260517).generate_state(1))
    seed0 = int(seeds[0])
    cfg = _cfg_real(traffic_source="milan", mobility_source="geolife",
                    milan_file="milan_2013-11-04.txt")
    out = {"_seed": seed0, "_milan_file": "milan_2013-11-04.txt",
           "_T_slots": cfg.time.T_slots}
    for ctl in CONTROLLERS_MAIN:
        t0 = time.time()
        res = run_episode(ctl, cfg, seed0, trace=True)
        out[ctl] = {"loss_trace": res["loss_trace"]}
        loss_arr = np.array(res["loss_trace"], dtype=float)
        print(f"  {ctl:>12}: mean={loss_arr.mean():.1f}  "
              f"p95={np.percentile(loss_arr, 95):.1f}  "
              f"p99={np.percentile(loss_arr, 99):.1f}  "
              f"({time.time() - t0:.1f}s)")
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, default=float)
    print(f"-> wrote {OUT_PATH}  ({os.path.getsize(OUT_PATH)/1024:.1f} kB)")


if __name__ == "__main__":
    main()
