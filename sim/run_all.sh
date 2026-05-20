#!/usr/bin/env bash
# Reproduce all experiments end-to-end.
# Usage:  ./sim/run_all.sh   (from project root)
set -e
cd "$(dirname "$0")/.."

mkdir -p sim/results figures

LOG=sim/results/run.log
echo "== run_all started $(date -Iseconds) ==" > "$LOG"

# Synthetic suite: main, V-sweep, sensing-sweep, risk-sweep, burst, blockage,
# convergence, ablation.
python3 -m sim.run all --workers 16 --seeds 10 2>&1 | tee -a "$LOG"

# Real-data suite (Milan Mon + Fri, Geolife): main, sensing, ablation,
# convergence; runs both days in a single invocation.
python3 -m sim.real_data_validation --workers 16 --seeds 10 2>&1 | tee -a "$LOG"

# Per-slot loss traces for the six controllers on Mon real data (single
# canonical seed) used by fig_delay_cdf.
python3 -m sim.run_main_traces 2>&1 | tee -a "$LOG"

# Remediation: Gamma calibration, sensing-wide sweep, ablation v2.
python3 -m sim.remediation --workers 16 --seeds 10 2>&1 | tee -a "$LOG"

# Reviewer-driven mismatch sweep (bias and sigma over-confidence) and its
# figure/table.
python3 -m sim.run_mismatch --workers 16 --seeds 10 2>&1 | tee -a "$LOG"
python3 -m sim.make_fig_mismatch 2>&1 | tee -a "$LOG"
python3 -m sim.make_table_mismatch 2>&1 | tee -a "$LOG"

# Figures and tables (covers all the .json outputs above).
python3 -m sim.make_figures 2>&1 | tee -a "$LOG"
python3 -m sim.make_tables  2>&1 | tee -a "$LOG"

echo "== run_all finished $(date -Iseconds) ==" | tee -a "$LOG"
