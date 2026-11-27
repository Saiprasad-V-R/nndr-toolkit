# Manuscript resources and reproducibility

Companion to *Inferring Non-Normal Amplification Geometry from Multivariate Time Series*, V. R. Saiprasad, V. Troude and D. Sornette (2026). The [original preprint](https://arxiv.org/abs/2607.14786) and the revised toolkit use different threshold conventions; see the [migration notes](../THRESHOLD_MIGRATION.md).

This repository is a reusable Python analysis package, with two self-contained synthetic validation experiments and one representative seizure EEG reproduction. The remaining empirical applications are documented below as methodological references, not as complete, independently reproduced revised pipelines. The original push-up recording is included. Large public datasets are not bundled.

## Software version

These instructions describe the reorganized checkout using the NNDR 2.0.0 scientific API. The published tag `2.0.0` retains its original directory layout. Follow the documentation within that snapshot if reproducing the fixed release. To obtain the current repository, clone it if needed:

```bash
git clone https://github.com/Saiprasad-V-R/nndr-toolkit.git
cd nndr-toolkit
```

The [historical implementation](https://github.com/Saiprasad-V-R/nndr-toolkit/tree/v1.0.0) is retained separately. For new analyses, use the revised release and record the software commit, preprocessing settings and random seeds.

## Environment

Run commands from the repository root in an activated Python environment (Python >= 3.9):

```bash
python -m pip install -e '.[dev]'
python -m pip install -r manuscript/seizure_eeg/requirements.txt
python -m pip check
```

The EEG dependency is optional if running synthetic experiments only. For the independently validated macOS Apple Silicon environment (Python 3.9.6, NumPy 1.26.4, SciPy 1.13.1, Matplotlib 3.9.4 and MNE 1.8.0), use [requirements-validated-macos.txt](../requirements-validated-macos.txt) as a constraints file. That file records one environment, not a universal cross-platform lockfile. See the [package README](../README.md) and [scientific API](../docs/scientific_api.md) for analysis of new data.

## Common revised methodology

For data with shape `(N_channels, T_samples)`, form observed adjacent pairs `X = x[:, :-1]`, `Y = x[:, 1:]` within each window. Centering, scaling and filtering are dataset-specific and must be stated. Relative ridge estimation and trace-preserving shrinkage are

```text
nu = nu_rel * tr(X X^T) / N
Fhat = Y X^T (X X^T + nu I)^(-1)
Freg = (1-alpha) Fhat + alpha * tr(Fhat)/N * I
Gamma = Q^T Freg Q
```

M2 maximizes directed coupling between orthogonal directions; M3 extracts the commutator plane and orders its input and reaction directions. Compare both on the same fitted operator. M1 is an eigenbasis-SVD baseline. Shrinkage does not guarantee stability. Windows are displayed at their trailing endpoint unless explicitly specified otherwise; offline zero-phase filtering is not causal preprocessing.

For a stable real 2-by-2 reduced map `Gamma`, the revised diagnostics are:

- `R_A`: variance-response criterion, using the real principal logarithm `L = log(Gamma)/dt`. For real rates `-gamma1, -gamma2`, set `Delta_rate = abs(gamma1-gamma2)/(gamma1+gamma2)` and `s = sqrt(1-Delta_rate^2)`; `Kc_A = sqrt(s/(1-s))`. For a complex pair `-gamma +/- i omega`, set `beta = abs(omega)/gamma` and `Kc_A = 1/sqrt(sqrt(1+beta^2)-1)`. Away from degeneracy, `R_A = K/Kc_A`. The implementation uses a finite repeated-eigenvalue limit where applicable. `R_A > 1` indicates a delayed increasing interval in the isotropic impulse variance-response kernel, not necessarily a peak exceeding its initial value.
- `R_sigma = d_F(Gamma)/sqrt((1-|lambda1|^2)(1-|lambda2|^2))`, where `d_F` is the Henrici departure from normality. For a stable reduced map, `R_sigma > 1` is equivalent to `sigma_max(Gamma) > 1`.
- `K = d_F(Gamma)/|lambda1-lambda2|` for distinct multipliers. Stability and principal-log admissibility are reported separately; undefined ratios are NaN. No forced spectral stabilization or clipping is used by the revised threshold implementation.
- The default package `support` is `||r||_1/(sqrt(N)||r||_2)`. Figure 12 deliberately retains its archived entropy support and discrete multiplier splitting, defined below. These are distinct diagnostics, not interchangeable names.

Neither ratio is a clinical, behavioral or causal threshold. Norm-dependent quantities depend on measurement coordinates. Agreement between M2 and M3 checks extraction-rule sensitivity, not the validity of the fitted operator. See [estimation and adequacy](../docs/estimation_adequacy.md) and [statistical controls](../docs/statistical_controls.md) for assumptions and limitations.

## 1. Synthetic validation (fully runnable)

These are newly designed checks of the revised mathematical framework, not exact recreations of numbered figures from the original preprint.

### 1.1 Real and complex threshold validation

```bash
python -W error::RuntimeWarning manuscript/synthetic/validate_thresholds.py --out results/synthetic/thresholds
```

The experiment evaluates 420 sampled stable maps from 140 real/complex generators, with independently computed kernel derivatives and direct singular values. The validated macOS run reported zero classification disagreements for either criterion, and 42 `R_A` boundary cases. The derivative oracle uses a finite numerical search, not an analytical proof.

![Real and complex threshold validation](synthetic/figures/threshold_ground_truth.png)

Outputs: `threshold_sweep.csv`, `summary.json`, `threshold_ground_truth.png` and `.svg`. See [experiment details](synthetic/README.md).

### 1.2 Finite-data recovery

```bash
python -W error::RuntimeWarning manuscript/synthetic/validate_recovery.py --out results/synthetic/recovery
```

A known two-dimensional block is embedded orthogonally in a stable 12-dimensional system. M1, M2 and M3 are compared on identical fitted operators at 200, 1000 and 5000 samples, for four target `R_A` values and twelve seeded realizations per spectral family and target. The default run produces 1,536 rows across truth, oracle, known-plane and inferred stages. Invalid estimates are retained in the raw results. At 5000 samples, M2/M3 achieved 100% `R_A` classification accuracy in this particular design, not a universal guarantee.

![Recovery benchmark](synthetic/figures/recovery_benchmark.png)

Outputs: `recovery_trials.csv`, `recovery_summary.csv`, `recovery_summary.json`, `recovery_benchmark.png` and `.svg`. See [experiment details](synthetic/README_recovery.md). The original benchmark's historical M3 support convention is documented there; it is not the current directed-support API. Included preview figures were generated in the original validation runs; use the commands above to regenerate them locally.

## 2. Representative seizure EEG (fully runnable)

**Data:** [CHB-MIT Scalp EEG Database, PhysioNet](https://physionet.org/content/chbmit/1.0.0/), recording `chb01/chb01_03.edf`, onset 2996 s and termination 3036 s. See [data attribution and license](seizure_eeg/DATA_LICENSE.md). The EDF is downloaded separately; the repository includes the derived plotting CSVs, original plotting scripts and reference PDFs.

**Reproduce the approved main and supplementary Figure 12 from archived CSVs:**

```bash
python manuscript/seizure_eeg/representative/scripts/plot_figure12_RA_main.py
python manuscript/seizure_eeg/representative/scripts/plot_figure12_chb01_03_Rsigma_supp.py
```

**Recompute from the original EDF and audit numerical parity:**

```bash
python manuscript/seizure_eeg/download_data.py --data-dir results/seizure/data
python -W error::RuntimeWarning manuscript/seizure_eeg/representative/evaluate_edf.py \
  --edf results/seizure/data/chb01_03.edf --out results/seizure/audit
```

The validated configuration uses a fixed core16 bipolar montage, 60 Hz notch (Q=30), fifth-order zero-phase Butterworth bandpass 0.5–15 Hz, anti-aliased polyphase resampling from 256 to 32 Hz, within-window centering, relative ridge 0.003 and shrinkage 0.05. Windows are 40 s long with 2 s step; 201 endpoints span -160 to +240 s relative to onset. M2 uses 12 restarts, 200 iterations and seed 0; M3 uses directed ordering. Only one window is entirely ictal. The figure-specific support is `exp(-sum(p_i log p_i))/N`, with `p_i = r_i^2/sum(r_j^2)`; splitting is `|lambda1-lambda2|/|lambda1+lambda2|`. Both figures retain the original four-panel style, raw traces and five-endpoint median overlays within valid segments. Original fixed plotting limits are retained; extreme values remain in the CSV.

The Mac audit passed all nine numerical comparisons with no finite-mask mismatches, and waveform maximum absolute error 1.28e-11 microvolts. Valid ratio counts were 190/201 for M2 and 201/201 for M3. The largest absolute ratio error was 1.62e-6. See the [parity report](seizure_eeg/representative/validation/parity_report.json), [representative README](seizure_eeg/representative/README.md), and [main](seizure_eeg/representative/reference_figures/figure12_chb01_03_RA_main.pdf) / [supplementary](seizure_eeg/representative/reference_figures/figure12_chb01_03_Rsigma_supp.pdf) reference figures. The evaluator reconstructs the original analysis within specified numerical tolerances; it is not the unavailable byte-identical original evaluation script. This is a descriptive single-event analysis, not a seizure-prediction validation. Cohort-level reproduction is outside this public example.

## 3. Additional empirical applications (methodological reference)

- [Uterine EHG](uterine_ehg/README.md): data source and historical analysis settings.
- [Freezing of gait](freezing_of_gait/README.md): data source and historical analysis settings.
- [Push-up recording](pushup/README.md): included recording and revised methodology.

These resources do not constitute independently verified complete revised empirical pipelines. The two synthetic validations above are newly designed checks of the revised framework; the older benchmark example in `synthetic/legacy_benchmark_example.py` is retained for historical context, not as a recommended new-user tutorial.

## Scope of validation and citation

Use the [main README](../README.md) for citation and software license. Public data have their own licenses and recommended citations. The selected synthetic and representative EEG examples were independently executed in the validated environment. The project does not claim that every figure, cohort result or statistical calibration in the revised manuscript is reproduced by this release. Model adequacy, surrogate assumptions, sampling aliasing and preprocessing must be assessed for each new application.
