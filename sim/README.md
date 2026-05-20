# Sensing-Aware Green RAN Simulator

Reference Python implementation backing the paper *Sensing-Aware Green RAN:
Lyapunov and Risk-Limited Control of Cell Sleep and Beam Management Using
ISAC-Derived Mobility and Blockage States* (TGCN submission).

## What it simulates

A `B=7` hexagonal cell layout (centre + 6-cell ring, ISD 200 m) at
28 GHz mmWave with 100 MHz bandwidth. Each cell exposes a downlink queue
driven by Poisson + Pareto-bursty arrivals; per-slot decisions choose
cell-sleep, a discrete beam from a 24-entry codebook (8 azimuths × 3 power
rungs), and a handover bias. The link uses 3GPP UMa-NLoS path loss, a
two-state Markov LoS/NLoS blockage model calibrated to TR 38.901 Block-A,
and Rayleigh small-scale fading. UE positions evolve via a bounded
random-waypoint mobility process. An EKF-based ISAC pipeline tracks each
UE's position/velocity from noisy range measurements (parameterized by
`sigma_isac_m`), exposing a posterior covariance whose trace drives the
Wasserstein ambiguity radius `eps = kappa * sqrt(tr(Sigma)) + kappa0`
used by the DR-CVaR Lyapunov controller.

## Files

| file                | purpose                                                     |
|---------------------|-------------------------------------------------------------|
| `config.py`         | All calibrated parameters (single source of truth).         |
| `channel.py`        | Path loss, blockage attenuation, BF gain, SINR computation. |
| `mobility.py`       | Random-waypoint generator producing UE positions/velocities.|
| `blockage.py`       | Two-state LoS/NLoS Markov model.                            |
| `traffic.py`        | Per-cell arrivals -- Milan trace if present, else synthetic.|
| `isac.py`           | EKF tracker + Wasserstein radius mapping.                   |
| `controllers.py`    | All six controllers (Always-On, Threshold, three Lyapunov   |
|                     | variants, Oracle DP).                                       |
| `metrics.py`        | Per-run summary + bootstrap CI.                             |
| `run.py`            | Episode driver, grid runner, CLI for experiments.           |
| `make_figures.py`   | Generates publication-grade PDFs.                           |
| `make_tables.py`    | Emits booktabs LaTeX tables.                                |
| `run_all.sh`        | End-to-end reproducer.                                      |

## Data sources

* **Milan Telecom dataset (2013-11-04)** — primary arrival-trace source.
  Expected location `sim/data/milan_2013-11-04.txt`, 8-column space-separated
  format `<cellID> <interval_ms> <country> <SMS_in> <SMS_out> <Call_in>
  <Call_out> <Internet>`. The simulator extracts the 7 most-active cell IDs,
  scales the *Internet* timeseries to a target mean of `base_rate_mbps` per
  cell, interpolates onto our slot grid, and adds Poisson plus
  Pareto-bursty noise.
* **Synthetic fallback**: if the Milan file is missing or unparseable, a
  per-cell diurnal sinusoid with Pareto bursts is used. The choice is
  reported as the first line of `run.log`.

## Reproducing all experiments

```bash
./sim/run_all.sh           # ~25 min on a 20-core machine
```

This runs (writing JSON to `sim/results/`):

| key            | description                                       |
|----------------|---------------------------------------------------|
| `exp_main`     | All six controllers at `V=1000`, `Gamma=100`.     |
| `exp_V`        | Paper's method swept over `V` in 10^2..10^5.       |
| `exp_sensing`  | Paper's method swept over `sigma_isac` (0.5..8 m).|
| `exp_risk`     | Paper's method swept over `Gamma`.                |
| `exp_burst`    | Five controllers vs burst probability.            |
| `exp_blockage` | Five controllers vs blockage probability.         |
| `exp_convergence` | Single long-run trace of q,z,tau (one seed).   |
| `exp_ablation` | Paper's method w/ vs w/o DR / virt-queue / sensing|

Figures are written to `figures/` and LaTeX tables to
`sim/results/table_*.tex`.

## Notes on units

* Queue `q_b` is in raw bits; the controller internally normalizes by
  `Q_SCALE = 5e6` (one-slot capacity scale) so that the Lyapunov knob `V`
  in the paper's 10^2..10^6 range maps to a meaningful sleep threshold.
* The CVaR loss proxy is `ell(t) = sum_b q_b / a_bar_b` (paper's
  definition). `Gamma` is dimensionless in the same scaling.

## Reproducibility

All randomness is seeded via `numpy.random.SeedSequence(20260517)`. Each
seed yields one independent episode; aggregate statistics use the 10
canonical seeds with 95% bootstrap CIs.
