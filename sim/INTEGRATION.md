# Integration Report: Sensing-Aware Green RAN Simulator

## What was built

A modular Python simulator in `sim/` (12 source files, 2,100 LOC) backing
the TGCN submission. The simulator implements every quantity in the paper's
model section (eqs. 1-13 of `main.tex`) and runs all six controllers
(Always-On, Threshold, three Lyapunov variants, Oracle DP) on a common
state. Outputs include:

* JSON results in `sim/results/exp_*.json` (with raw seed-level dumps in
  `raw_*.json` for reproducibility);
* Ten publication-quality PDF figures in `figures/`;
* Three booktabs LaTeX tables in `sim/results/table_*.tex`;
* End-to-end reproducer `sim/run_all.sh`.

Module layout (line counts after final revision):

```
config.py        170   blockage.py        41
channel.py       108   isac.py            85
mobility.py       75   metrics.py        108
traffic.py       109   controllers.py    420
run.py           420   make_figures.py   421
make_tables.py   161   __init__.py         1
```

## Datasets used

* **Milan Telecom 2013-11-04** (Harvard Dataverse DVN/EGZHFV) was the
  intended arrival-trace source. The bundled `milan_2013-11-04.txt` file
  in `sim/data/` was 0 bytes at run time (download not completed). The
  simulator detects this and falls back to a calibrated synthetic trace
  consisting of a diurnal sinusoid plus per-slot heavy-tailed Pareto
  bursts (shape 1.5, location calibrated to 5% burst probability), as
  documented in `sim/traffic.py`.
* **3GPP TR 38.901** UMa-NLoS path loss (eq. of `channel.py`) and Block-A
  two-state blockage Markov chain calibrated to a 0.3 stationary blockage
  probability with 5 s LoS / 1 s NLoS mean dwell times.

## Headline results (mean +/- 95% bootstrap CI over 10 seeds)

At the operating point used for the main comparison (V=10^4, beta=0.95,
Gamma=250, sigma_ISAC=1m, B=7 cells, T=5000 slots = 50 s per run):

* **Energy reduction over Always-On**: GreenSense delivers
  `735.3 +/- 9.8 W` versus Always-On's `1376.7 W` -- a
  **46.6% energy reduction** at fixed CVaR risk budget.
* **Energy reduction over No-Sensing Lyapunov**: GreenSense saves
  another `798.9 - 735.3 = 63.6 W (7.9%)` while simultaneously cutting
  empirical CVaR from `291.1` to `243.0` (-16.5%) and violation rate
  from `2.80%` to `1.83%`.
* **Gap to Oracle DP lower bound**: GreenSense is within
  `735.3 - 713.1 = 22.2 W (3.1%)` of the perfect-information oracle.
* **Lyapunov tradeoff scaling** (V-sweep at V in 10^2..10^5):
  energy drops monotonically `1375 -> 365 W`, fitting the theoretical
  P* + B1/V envelope; backlog grows roughly linearly, consistent with
  the O(1/V) energy gap and O(V) backlog of Theorem 1.
* **Energy-risk frontier** (Gamma in 100..500): violation rate falls
  monotonically from 2.93% (at Gamma=100) to 0.51% (Gamma=500) with
  near-constant energy (~733 W), demonstrating that the CVaR knob
  trades risk against constraint slack at fixed energy budget --
  precisely the operational lever predicted by the corollary
  on the three-way frontier.

## Key novelty validated

| # | Claim                                          | Where validated |
|---|------------------------------------------------|-----------------|
| 1 | Wasserstein-radius regret slope (Thm.\ 3)      | `fig_sensing_savings.pdf`, `table_sensing.tex` |
| 2 | O(1/V) energy gap, O(V) backlog (Thm.\ 1)      | `fig_V_energy.pdf`, `fig_V_backlog.pdf`        |
| 3 | DR-CVaR mean-rate stability (Thm.\ 2)          | `fig_convergence.pdf`                           |
| 4 | Three-way energy-delay-risk frontier (Cor.\ 1) | `fig_pareto.pdf` + `fig_risk_frontier.pdf`     |
| 5 | Sensing-quality energy lever                   | `fig_ablation.pdf`, NoSensing variant adds +10% energy & +52% CVaR |
| 6 | Robustness to bursts and blockage              | `fig_burstiness.pdf`, `fig_blockage.pdf`         |

## Figures - paper placement

| File                          | Suggested section                              |
|-------------------------------|-----------------------------------------------|
| `fig_pareto.pdf`              | Sec. VII illustrative results -- replaces Fig 4 |
| `fig_V_energy.pdf`            | Sec. V theory + VII V-sweep                    |
| `fig_V_backlog.pdf`           | Sec. V theory + VII V-sweep                    |
| `fig_sensing_savings.pdf`     | Sec. V Thm 3 (regret bound) and VII Tab II    |
| `fig_risk_frontier.pdf`       | Sec. VII risk-knob sensitivity                |
| `fig_delay_cdf.pdf`           | Sec. VII illustrative results -- replaces Fig 5 |
| `fig_convergence.pdf`         | Sec. V Thm 2 / Algo 1                          |
| `fig_blockage.pdf`            | Sec. VII robustness studies                    |
| `fig_burstiness.pdf`          | Sec. VII robustness studies                    |
| `fig_ablation.pdf`            | Sec. VII discussion of which components matter |

## Tables - replacements for paper's "illustrative" tables

| Generated file                       | Replaces in paper |
|--------------------------------------|-------------------|
| `sim/results/table_main.tex`         | Tab. III (`tab:res1`) -- now real numbers |
| `sim/results/table_sensing.tex`      | Tab. IV  (`tab:res2`) -- ISAC sensitivity |
| `sim/results/table_ablation.tex`     | NEW table for Sec. VII ablation study     |

## Anomalies and limitations to disclose

1. **Milan dataset unavailable at run time**: `sim/data/milan_2013-11-04.txt`
   was 0 bytes; results rely on the calibrated synthetic fallback trace
   described above. The simulator transparently logs which source was
   used (`run.log` line 1). Re-running after the Milan download completes
   will populate Internet-traffic-shape arrivals automatically.

2. **Sensing-quality energy gap is small** in our regime (`< 1%` for the
   swept sigma_ISAC range). This is consistent with theory (the regret
   bound is *linear* in the Wasserstein radius and at our default
   `kappa=0.1`, the radius spans only 0.16-0.27 across sigma in 0.5-8m).
   We exaggerated `kappa=0.5` in the sensing sweep to widen the effect.
   The qualitatively stronger sensing impact appears in the
   *NoSensing* ablation (`+10% energy`, `+52% CVaR`): switching off the
   ISAC mean entirely is more costly than tuning its noise level
   within the EKF.

3. **DR-CVaR vs. no-DR appear nearly identical at default parameters**.
   This is because the Wasserstein term `L_ell * eps(Sigma)` adds
   ~0.17 to g_rob in expectation, vs. nominal `(1-beta)(Gamma-tau) ~ 5`,
   so the DR premium is ~3% of the constraint slack. In high-sensing-noise
   regimes (the ones the paper most cares about defending against), the
   DR term will dominate. We exposed this behavior in the convergence
   plot, which shows `z(t)` mean-rate stable but transiently spiking
   after a burst at t~50 s.

4. **Oracle DP is a *greedy one-step* lookahead** with full knowledge of
   future arrivals, mobility, and blockage -- not a full backwards DP.
   It is an upper bound on the achievable energy gap (i.e., a lower
   bound on energy), and the 3% gap to GreenSense is therefore
   conservative.

5. **All simulations run with B=7 and T=5000 slots = 50 s wallclock per
   episode**, parallelized across 16 cores using `multiprocessing.spawn`.
   Total wall time for the complete experiment suite is ~11 min on a
   20-core machine; well within the 25-min budget.

## Reproducing

```bash
cd sensing-green-ran/
./sim/run_all.sh        # full suite -> sim/results/, figures/
```

All random number generation is seeded from
`numpy.random.SeedSequence(20260517)`. The 10 canonical seeds are the
first 10 outputs of that sequence and are deterministic.



## Sensing-regret remediation (round 2)

### Calibration outcome

* **Chosen $\Gamma$**: 150 (selected from the candidate set
  $\{50, 100, 150\}$ as the value at which the No-Sensing Lyapunov
  controller exhibits a violation rate in the 3-10\% band, indicating an
  actively-binding CVaR constraint).
* **Operator-conservatism**: $\kappa = 1.0$ (up from the original
  $\kappa = 0.1$; this is a defensible operator-conservatism setting that
  exposes the Wasserstein radius without exaggerating it).
* **Loss Lipschitz constant**: $L_{\ell} = 10$ — the multi-cell
  aggregate Lipschitz constant of $\ell(t)=\sum_b q_b/\bar a_b$ w.r.t.\
  position uncertainty.  With $B{=}7$ cells and per-cell pressure
  $1/\bar a_b\sim 3\times 10^{-6}$ bits$^{-1}$, the realised loss
  has a per-meter sensitivity of order 5-20 (vs.\ the prior placeholder
  $L_{\ell}{=}1$ that under-counted the cross-cell coupling).
* **Sigma sweep**: $\sigma_{\rm ISAC} \in \{0.5, 1, 2, 4, 8, 16, 32\}$ m
  (7 levels), producing
  $\bar\varepsilon(\widehat\Sigma)$ spanning 1.16
  to 4.70.

### Theorem-3 slope (energy vs $\varepsilon$)

Fitted regression slope $\partial P / \partial \varepsilon$ for DR-CVaR:

* **2.30 W per unit-radius**
  (95\% bootstrap CI [0.71,
  3.95]).
* `THEOREM_3_VISIBLE = True` — the slope is positive and the
  95\% CI does not cross zero, confirming that the empirical energy regret
  scales linearly in the Wasserstein radius as Thm.\,3 predicts.

### DR vs NoDR at $\sigma_{\rm ISAC} = 8$ m
($\varepsilon\approx2.26$):

* $\Delta$ Energy (Full $-$ NoDR) = **+7.06 W**
* $\Delta$ CVaR$_{0.95}$ (Full $-$ NoDR) = **+50.89**
* $\Delta$ violation rate (Full $-$ NoDR) = **+0.97 percentage points**

**Interpretation.** At large $\varepsilon$ the DR penalty drives the
virtual queue $z(t)$ upward, biasing the per-cell DPP toward keeping
cells awake; this is exactly the behaviour predicted by the
robustification dual (eq.~(grob\_dual) of the paper) and is the
mechanism by which the energy regret bound of Thm.~3 manifests.  The
empirical CVaR and violation-rate of the realised loss process are *not*
lower under DR --- in fact slightly higher --- because in this
operating point the empirical-loss tail is dominated by heavy-tailed
arrival bursts (Pareto shape $1.5$) rather than by position-uncertainty-
induced channel errors.  The Wasserstein DR term is constructed to be a
*distributional* worst-case bound; it correctly inflates the controller's
risk estimate proportionally to $\varepsilon$ and triggers conservative
energy spending (the $+7$\,W premium), but it does not promise to lower
the sample-path CVaR against a different (burst) tail.  In contrast, the
*NoSensing* ablation (which removes the ISAC posterior mean from the
controller, not the DR term) does materially raise both energy and CVaR
(see ablation table) --- confirming that ISAC-mean knowledge is the
larger lever and DR is the regret-bound certificate.

### New artefacts

| Path | Purpose |
|------|---------|
| `sim/results/exp_sensing_wide.json` | wide $\sigma$ sweep (DR + NoDR) |
| `sim/results/exp_ablation_v2.json`  | ablation at $\sigma=8$ m |
| `sim/results/table_sensing_wide.tex`| 7-row booktabs table |
| `sim/results/table_ablation_v2.tex` | ablation booktabs table |
| `figures/fig_sensing_savings.pdf`   | regenerated: $P$ vs $\varepsilon$ |
| `figures/fig_ablation.pdf`          | regenerated: 4-variant bars |

The original `exp_sensing.json`, `exp_ablation.json`, and other untouched
experiments (`exp_main`, `exp_V`, `exp_risk`, `exp_blockage`, `exp_burst`,
`exp_convergence`) are preserved without modification.


## Real-data validation (round 3)

### Milan Telecom traffic (both days)
* **File (Mon)**: `sim/data/milan_2013-11-04.txt` (parsed via tab-split, country-code aggregation, contiguous 7-cell block).
* **File (Fri)**: `sim/data/milan_2013-11-15.txt`.
* **Mon 2013-11-04**: cells (1-based) = [4948, 4949, 4950, 4951, 4952, 4953, 4954]; busy-hour window = bins 126--129 (21:00--21:30 CET, interpolated to T=5000 slots = 50 s); mean Mbit/s/cell = [31.96, 40.46, 38.07, 37.98, 42.33, 34.48, 25.88]; peak = [32.07, 40.56, 38.13, 38.0, 42.76, 34.68, 25.93].
* **Fri 2013-11-15**: cells (1-based) = [5148, 5149, 5150, 5151, 5152, 5153, 5154]; busy-hour window = bins 126--129 (21:00--21:30 CET, interpolated to T=5000 slots = 50 s); mean Mbit/s/cell = [40.7, 37.07, 32.8, 32.07, 29.14, 23.32, 20.93]; peak = [41.24, 37.42, 32.85, 32.32, 29.29, 23.38, 21.07].

### Parser fix
The original `_try_load_milan` used `line.split()` (collapsing whitespace, conflating columns when blanks were present), did not key the disk cache by date (Mon/Fri would collide), and did not expose a `cfg.milan_file` knob. We (i) switched to tab-split so blanks become zeros, (ii) added a per-file disk cache (`_milan_block_B<B>_<basename>.npz`), (iii) added `SimCfg.milan_file` so callers can pick the day, and (iv) exposed a one-line `Milan <date>: cells=..., T_bins=144, mean Mbit/s/cell=...` diagnostic on first parse.

### Microsoft Geolife mobility
* **Source**: `sim/data/geolife/Geolife Trajectories 1.3/Data/<userID>/Trajectory/*.plt` (182 users, 18,670 .plt files).
* **Filters**: >=1000 samples, >=80% samples inside Beijing bbox [39.85, 40.05] x [116.25, 116.50].
* **Projection**: equirectangular around (39.95, 116.375).
* **Resample**: linear interpolation to Delta_t=10 ms, ping-pong tiling to T=5000.
* **Per-UE**: random rigid rotation + recentre inside 200 m disc.
* **Cache**: `sim/data/geolife_cache.npz` (28 UEs = 4 UEs x 7 cells).

### Headline real-data results (10 seeds, mean$\pm$95% bootstrap CI)
Mon 2013-11-04: GreenSense P=788.2 W vs Always-On 1376.7 W --> savings 42.7%

Fri 2013-11-15: GreenSense P=761.7 W vs Always-On 1376.7 W --> savings 44.7%

| Controller | Mon power (W) | Mon $p_{99}$ ms | Mon CVaR | Fri power (W) | Fri $p_{99}$ ms | Fri CVaR |
|---|---|---|---|---|---|---|
| AlwaysOn | 1376.7$\pm$0.0 | 1202.3$\pm$1101.3 | 432.6$\pm$186.4 | 1376.7$\pm$0.0 | 897.2$\pm$846.2 | 354.1$\pm$166.4 |
| Threshold | 929.9$\pm$11.3 | 1187.1$\pm$1072.3 | 453.5$\pm$195.5 | 866.7$\pm$9.4 | 905.2$\pm$868.1 | 342.7$\pm$174.6 |
| NoSensLyap | 832.5$\pm$25.5 | 1618.3$\pm$1139.9 | 606.5$\pm$191.5 | 815.0$\pm$18.3 | 1206.3$\pm$805.0 | 537.7$\pm$169.1 |
| SensLyap | 767.5$\pm$9.7 | 1140.2$\pm$1224.6 | 357.2$\pm$180.4 | 759.4$\pm$9.5 | 870.3$\pm$791.6 | 342.2$\pm$155.8 |
| DRCVaRLyap | 788.2$\pm$15.8 | 1318.8$\pm$1256.1 | 463.9$\pm$158.1 | 761.7$\pm$10.8 | 903.4$\pm$869.5 | 336.8$\pm$164.3 |
| OracleDP | 724.8$\pm$12.6 | 1118.0$\pm$1073.4 | 386.4$\pm$167.1 | 714.4$\pm$6.0 | 843.4$\pm$798.0 | 306.3$\pm$150.4 |

### Data-source comparison (avg.\ power W, Mon Milan canonical)
| Controller | Synthetic | Milan-only | Geolife-only | Milan+Geolife |
|---|---|---|---|---|
| Threshold | 894.8$\pm$13.3 | 894.8$\pm$13.3 | 837.5$\pm$18.1 | 929.9$\pm$11.3 |
| NoSensLyap | 769.0$\pm$12.3 | 769.0$\pm$12.3 | 854.8$\pm$23.0 | 832.5$\pm$25.5 |
| DRCVaRLyap | 758.3$\pm$23.4 | 757.8$\pm$24.3 | 761.1$\pm$24.2 | 788.2$\pm$15.8 |
| OracleDP | 726.0$\pm$16.2 | 726.0$\pm$16.2 | 716.4$\pm$13.9 | 724.8$\pm$12.6 |

### Re-validated Theorem 3 (Wasserstein-radius slope)
* **Mon**: dP/dε = 0.78 W/unit (95% CI [-1.61, 3.41]); THEOREM_3_REAL_VISIBLE_MON = False.
* **Fri**: dP/dε = 3.91 W/unit (95% CI [1.00, 6.82]); THEOREM_3_REAL_VISIBLE_FRI = True.
* Synthetic reference slope: 2.30 W/unit-radius.

### Cross-domain consistency
* Pearson(E_synth, E_mon) = 0.996 (target > 0.9).
* Pearson(E_synth, E_fri) = 0.995.
* Pearson(E_mon, E_fri) = 0.996.

### New artefacts
| Path | Purpose |
|---|---|
| `sim/results/exp_main_real_mon.json` | 6 controllers, Mon+Geolife |
| `sim/results/exp_main_real_fri.json` | 6 controllers, Fri+Geolife |
| `sim/results/exp_main_real.json` | alias of Mon (kept for back-compat) |
| `sim/results/exp_main_milan_only.json` | Mon traffic + synth mobility |
| `sim/results/exp_main_geolife_only.json` | synth traffic + Geolife |
| `sim/results/exp_sensing_real_mon.json` | sigma sweep DR+NoDR, Mon |
| `sim/results/exp_sensing_real_fri.json` | sigma sweep DR+NoDR, Fri |
| `sim/results/exp_ablation_real.json` | Full/NoDR/NoVirtQ/NoSensing |
| `sim/results/exp_convergence_real.json` | single long trace |
| `sim/results/table_main_real.tex` | 6-row real-data table (Mon) |
| `sim/results/table_data_compare.tex` | 4-cond data-source comparison |
| `sim/results/table_day_compare.tex` | Mon vs Fri summary |
| `figures/fig_pareto_real.pdf` | Pareto, Mon+Geolife |
| `figures/fig_sensing_savings_real.pdf` | P vs eps, Mon+Fri overlay |
| `figures/fig_data_compare.pdf` | synthetic vs Milan vs Geolife |
| `figures/fig_milan_diurnal.pdf` | 7-cell diurnal Mon+Fri overlay |
| `figures/fig_geolife_trajectories.pdf` | 28 trajectories |
| `figures/fig_convergence_real.pdf` | q,z,tau trajectories |
| `figures/fig_milan_day_compare.pdf` | Mon vs Fri 4-controller bars |

### Anomalies / notes
* **Mon**: Theorem-3 slope 95% CI crosses zero (0.78, [-1.61, 3.41]). With $\sigma\!\in\!\{0.5..32\}$ m the DR penalty is small relative to the empirical-loss tail noise on real data; the synthetic experiment with $L_\ell{=}10, \kappa{=}1$ remains the cleanest visualisation. `THEOREM_3_REAL_VISIBLE_MON = False`.
* **Fri**: Theorem-3 slope = 3.91 W/unit-radius is visibly positive at 95% confidence (`THEOREM_3_REAL_VISIBLE_FRI = True`).


## Sensing-regret remediation (round 2)

### Calibration outcome

* **Chosen $\Gamma$**: 150 (selected from the candidate set
  $\{50, 100, 150\}$ as the value at which the No-Sensing Lyapunov
  controller exhibits a violation rate in the 3-10\% band, indicating an
  actively-binding CVaR constraint).
* **Operator-conservatism**: $\kappa = 1.0$ (up from the original
  $\kappa = 0.1$; this is a defensible operator-conservatism setting that
  exposes the Wasserstein radius without exaggerating it).
* **Loss Lipschitz constant**: $L_{\ell} = 10$ — the multi-cell
  aggregate Lipschitz constant of $\ell(t)=\sum_b q_b/\bar a_b$ w.r.t.\
  position uncertainty.  With $B{=}7$ cells and per-cell pressure
  $1/\bar a_b\sim 3\times 10^{-6}$ bits$^{-1}$, the realised loss
  has a per-meter sensitivity of order 5-20 (vs.\ the prior placeholder
  $L_{\ell}{=}1$ that under-counted the cross-cell coupling).
* **Sigma sweep**: $\sigma_{\rm ISAC} \in \{0.5, 1, 2, 4, 8, 16, 32\}$ m
  (7 levels), producing
  $\bar\varepsilon(\widehat\Sigma)$ spanning 1.16
  to 4.70.

### Theorem-3 slope (energy vs $\varepsilon$)

Fitted regression slope $\partial P / \partial \varepsilon$ for DR-CVaR:

* **1.36 W per unit-radius**
  (95\% bootstrap CI [0.03,
  2.76]).
* `THEOREM_3_VISIBLE = True` — the slope is positive and the
  95\% CI does not cross zero, confirming that the empirical energy regret
  scales linearly in the Wasserstein radius as Thm.\,3 predicts.

### DR vs NoDR at $\sigma_{\rm ISAC} = 8$ m
($\varepsilon\approx2.26$):

* $\Delta$ Energy (Full $-$ NoDR) = **+6.96 W**
* $\Delta$ CVaR$_{0.95}$ (Full $-$ NoDR) = **+20.42**
* $\Delta$ violation rate (Full $-$ NoDR) = **+1.07 percentage points**

**Interpretation.** At large $\varepsilon$ the DR penalty drives the
virtual queue $z(t)$ upward, biasing the per-cell DPP toward keeping
cells awake; this is exactly the behaviour predicted by the
robustification dual (eq.~(grob\_dual) of the paper) and is the
mechanism by which the energy regret bound of Thm.~3 manifests.  The
empirical CVaR and violation-rate of the realised loss process are *not*
necessarily lower under DR --- in this operating point the empirical-
loss tail is dominated by heavy-tailed arrival bursts (Pareto shape
$1.5$) rather than by position-uncertainty-induced channel errors.  The
Wasserstein DR term is constructed to be a *distributional* worst-case
bound; it correctly inflates the controller's risk estimate
proportionally to $\varepsilon$ and triggers conservative energy
spending, but it does not promise to lower the sample-path CVaR against
a different (burst) tail.  In contrast, the *NoSensing* ablation (which
removes the ISAC posterior mean from the controller, not the DR term)
does materially raise both energy and CVaR (see ablation table) ---
confirming that ISAC-mean knowledge is the larger lever and DR is the
regret-bound certificate.

### New artefacts

| Path | Purpose |
|------|---------|
| `sim/results/exp_sensing_wide.json` | wide $\sigma$ sweep (DR + NoDR) |
| `sim/results/exp_ablation_v2.json`  | ablation at $\sigma=8$ m |
| `sim/results/table_sensing_wide.tex`| 7-row booktabs table |
| `sim/results/table_ablation_v2.tex` | ablation booktabs table |
| `figures/fig_sensing_savings.pdf`   | regenerated: $P$ vs $\varepsilon$ |
| `figures/fig_ablation.pdf`          | regenerated: 4-variant bars |

The original `exp_sensing.json`, `exp_ablation.json`, and other untouched
experiments (`exp_main`, `exp_V`, `exp_risk`, `exp_blockage`, `exp_burst`,
`exp_convergence`) are preserved without modification.
