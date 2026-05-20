# Sensing-Aware Green RAN

Reference simulator, experiment configurations, and reproducible numerical
results for the paper

> **Sensing-Aware Green RAN via Wasserstein-Robust Lyapunov Control with
> Tail-Reliability Guarantees**
> Liang Dong, Senior Member, IEEE.
> Submitted to *IEEE Transactions on Green Communications and Networking*.

The codebase implements a per-slot controller that minimizes a Lyapunov
drift-plus-penalty objective augmented with a virtual CVaR queue, with the
Wasserstein ambiguity radius calibrated to the posterior covariance of an
EKF-based ISAC pipeline. The deployment maps to a near-RT O-RAN xApp with
non-RT rApp hyperparameter tuning.

## Headline results

Trace-driven simulation on Milan Telecom traffic (Mon 2013-11-04, Fri
2013-11-15) combined with Microsoft Geolife GPS mobility, ten canonical
seeds, 95% bootstrap confidence intervals:

| Metric | Value |
|---|---|
| Energy reduction vs Always-On (Mon) | **42.7 %** (788.2 ± 15.8 W vs 1376.7 W) |
| Energy reduction vs Always-On (Fri) | **44.7 %** (761.7 ± 10.8 W vs 1376.7 W) |
| Gap to Oracle DP (clairvoyant one-step) | **6 – 9 %** |
| Cross-day Pearson correlation | **0.996** |
| Sensing-regret slope (Friday, Theorem 3) | **3.91 W per unit Wasserstein radius**, 95 % CI [1.00, 6.82] |

## Repository layout

```
sim/
├── config.py                Calibrated parameters (single source of truth)
├── channel.py               Path loss, blockage, beamforming gain, SINR
├── mobility.py              Random-waypoint synthetic mobility
├── mobility_geolife.py      Real Geolife trajectory projection
├── geolife_loader.py        Geolife data parser/cache
├── blockage.py              Two-state LoS/NLoS Markov model
├── traffic.py               Per-cell arrivals (Milan trace if present, else synthetic)
├── isac.py                  EKF tracker + Wasserstein radius mapping
├── controllers.py           Six controllers: Always-On, Threshold, three Lyapunov variants, Oracle DP
├── metrics.py               Per-run summary statistics with bootstrap CI
├── run.py                   Episode driver, grid runner, synthetic-suite CLI
├── real_data_validation.py  Mon + Fri real-data experiments end-to-end
├── remediation.py           Gamma calibration, sensing-wide sweep, ablation v2
├── run_mismatch.py          Posterior-bias and covariance-underestimate sweeps
├── run_main_traces.py       Single-seed per-slot loss traces for CCDF figure
├── make_figures.py          Publication-quality PDF figures
├── make_tables.py           booktabs LaTeX tables
├── make_fig_mismatch.py     Mismatch-experiment figure
├── make_table_mismatch.py   Mismatch-experiment table
├── INTEGRATION.md           Detailed numerical-results report
├── run_all.sh               End-to-end reproducer
├── data/                    (gitignored) place Milan + Geolife data here
└── results/                 (committed) raw JSONs + LaTeX table snippets
figures/                     Rendered PDFs (regenerable from results/)
```

## Requirements

* Python 3.10+
* `numpy`, `scipy`, `matplotlib`
* About 16 CPU cores recommended for the full Monte Carlo (each run completes
  in seconds; the full suite finishes in about 15 minutes on 16 cores).
* About 1 GB of free disk (raw Milan traces are 360 MB each; Geolife archive
  is roughly 800 MB).

```bash
pip install numpy scipy matplotlib
```

No GPU dependencies; everything runs on CPU.

## Data sources (not redistributed)

Two real public datasets are used. Neither is bundled in this repository;
the loaders look for them at the paths below.

### Milan Telecom 2013 — traffic

Download the two-weekday Internet activity traces from the Telecom Italia
Big Data Challenge, Harvard Dataverse:

> Barlacchi et al., "A multi-source dataset of urban life in the city of
> Milan and the Province of Trentino," *Scientific Data* 2, art. no. 150055,
> 2015. Dataverse DOI: [10.7910/DVN/EGZHFV](https://doi.org/10.7910/DVN/EGZHFV)

After accepting the dataset terms, place the following two files in
`sim/data/`:

```
sim/data/milan_2013-11-04.txt    # Monday, ~361 MB
sim/data/milan_2013-11-15.txt    # Friday, ~365 MB
```

The traffic loader (`sim/traffic.py`) automatically selects the 7 most-active
cell IDs from the file, scales the *Internet* timeseries to the per-cell
target rate, and interpolates onto the 10 ms slot grid.

### Microsoft Geolife — mobility

Download the Geolife Trajectories 1.3 archive from Microsoft Research:

> Zheng et al., "GeoLife: A collaborative social networking service among
> user, location and trajectory," *IEEE Data Engineering Bulletin* 33(2):
> 32-39, 2010. Available from
> [microsoft.com/en-us/research/publication/geolife-gps-trajectory-dataset-user-guide](https://www.microsoft.com/en-us/research/publication/geolife-gps-trajectory-dataset-user-guide/).

Extract so the file structure is:

```
sim/data/geolife/Geolife Trajectories 1.3/Data/000/Trajectory/*.plt
sim/data/geolife/Geolife Trajectories 1.3/Data/001/Trajectory/*.plt
...
```

The loader (`sim/geolife_loader.py`) caches a 28-trajectory subset filtered
to the Beijing bounding box (39.85 - 40.05 N, 116.25 - 116.50 E); the cache
file `sim/data/geolife_cache.npz` is regenerated automatically on first run.

If either dataset is absent, the simulator falls back to a calibrated
synthetic alternative; this is reported in stdout and the simulator's stored
metadata so reproductions remain unambiguous.

## Quickstart

Run the full synthetic suite (no external data required):

```bash
python3 -m sim.run all --workers 16 --seeds 10
python3 -m sim.make_figures
python3 -m sim.make_tables
```

Outputs land in `sim/results/exp_*.json` and `figures/fig_*.pdf`.

## Full paper reproduction

To regenerate every number, table, and figure in the paper from scratch
(requires Milan + Geolife data installed as above; about 15 minutes on 16
cores):

```bash
./sim/run_all.sh
```

This script runs in order:

1. `sim.run all` — synthetic suite (`exp_main.json`, `exp_V.json`,
   `exp_sensing.json`, `exp_risk.json`, `exp_burst.json`, `exp_blockage.json`,
   `exp_convergence.json`, `exp_ablation.json`).
2. `sim.real_data_validation` — Milan Mon + Fri main results, sensing
   sweep on both days, real-data ablation, real-data convergence.
3. `sim.run_main_traces` — single-seed per-slot loss traces for the CCDF
   figure (Mon canonical seed).
4. `sim.remediation` — synthetic Gamma calibration, wide sensing sweep
   at `Gamma=150`, ablation v2 at `sigma=8 m`.
5. `sim.run_mismatch` — posterior-bias and covariance-underestimate sweeps
   on Mon Milan + Geolife.
6. `sim.make_fig_mismatch`, `sim.make_table_mismatch`,
   `sim.make_figures`, `sim.make_tables` — all rendered outputs.

The canonical seed sequence is `numpy.random.SeedSequence(20260517)`. The
first generated seed `1252099755` drives single-seed traces; the ten-seed
Monte Carlo uses all ten generated seeds.

## Configuration knobs

All defaults live in `sim/config.py`. Key parameters:

| Group | Field | Default | Notes |
|---|---|---|---|
| Topology | `B` | 7 | Number of cells (center + ring) |
| Topology | `isd_m` | 200 | Inter-site distance, m |
| Time | `dt_ms` | 10 | Slot duration, ms |
| Time | `T_slots` | 5000 | Slots per episode (= 50 s) |
| Control | `V` | 10 000 | Lyapunov tradeoff (sweepable) |
| Control | `beta` | 0.95 | CVaR confidence |
| Control | `Gamma` | 250 | CVaR budget (paper headline uses 150) |
| Control | `loss_max` | 1000 | Cap in the state-dependent loss |
| Control | `alpha_tau` | 3e-4 | Robbins-Monro step base |
| Control | `alpha_tau_decay` | 1e-3 | R-M step: `c / (1 + t*decay)` |
| ISAC | `sigma_isac_m` | 1.0 | Position measurement std, m |
| ISAC | `kappa` | 0.10 | Wasserstein radius slope |
| ISAC | `kappa0` | 0.05 | Residual radius |
| ISAC | `L_loss_Lipschitz` | 1.0 | L_ell (paper headline uses 10) |
| Traffic | `base_rate_mbps` | 30 | Per-cell mean (synthetic + Milan-rescaled) |
| Traffic | `burst_prob` | 0.05 | Pareto-burst incidence |

The synthetic-versus-real overrides are explicit in the relevant runner
scripts (`real_data_validation.py`, `remediation.py`, `run_mismatch.py`).

## Citation

If you use this code, please cite the paper:

```bibtex
@article{dong2026sensinggreen,
  title   = {Sensing-Aware Green {RAN} via Wasserstein-Robust Lyapunov
             Control with Tail-Reliability Guarantees},
  author  = {Dong, Liang},
  journal = {IEEE Trans. Green Commun. Netw.},
  year    = {2026},
  note    = {Under review}
}
```

## Author

Liang Dong, Senior Member, IEEE
Department of Electrical and Computer Engineering
Baylor University, Waco, TX
[liang_dong@baylor.edu](mailto:liang_dong@baylor.edu)

## License

MIT, see [`LICENSE`](LICENSE). The two real datasets remain subject to their
upstream licenses (Harvard Dataverse terms for Milan Telecom; Microsoft
Research license for Geolife).
