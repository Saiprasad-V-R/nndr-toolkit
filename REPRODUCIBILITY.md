# Reproducibility

This document records, for each empirical dataset in the paper *Inferring
Non-Normal Amplification Geometry from Multivariate Time Series* (Saiprasad,
Troude & Sornette, 2026), the exact data source, the selected recordings and
channels, the preprocessing, the moving-window parameters, the analysis steps,
and the figure-generation workflow.

## Scope and design

This repository is released as a **reusable calibration and analysis toolkit**
(the `nndr` package). The empirical studies in the paper each required a
different, dataset-specific workflow: the seizure recordings alone are ~42 GB
and were downloaded and analysed directly on the Risks-X server, the EHG data
needed dataset-specific handling of annotations, channels, and recordings, and
some final figures were prepared separately. Rather than ship each large raw
dataset, this file documents every step precisely enough to reproduce the
analyses from the public sources.

Three of the four empirical datasets are **public** (Icelandic EHG, CHB-MIT
seizure EEG, Daphnet freezing of gait). The fourth (push-up inertial recording)
is an original recording and is **included in this repository** under
[`data/pushup/`](data/pushup/).

All four empirical analyses use the common moving-window construction of
Section 6.2 of the paper: fit a local one-step operator `F̂ⱼ` by
ridge-regularized regression inside each window, apply optional trace-preserving
isotropic shrinkage `α`, extract a two-dimensional response plane by two
independent methods (M2, optimization-based directed coupling; M3, commutator),
and compute the reduced diagnostics `R = K/Kc(Δ)`, the reaction support `S(r̂)`,
and the reduced eigenvalue splitting `Δ`. Every diagnostic is displayed at the
**window end time** (causal/trailing convention), so no sample later than
`t_end` contributes to the value shown at `t_end`.

The reduced diagnostics themselves (`Δ`, `κ2D`, `K`, `Kc(Δ)`, `R`, `S(r̂)`) are
computed exactly as in the packaged toolkit (`nndr.reduce2d`, `nndr.core`) and
match Eqs. (14)–(18) and Appendix A/B of the paper. The per-dataset scripts are
standalone (they re-implement the same M1/M2/M3 extraction so each can run
without importing the package); results are numerically consistent with the
`nndr` package API.

## Environment

```bash
python >= 3.9
pip install numpy scipy pandas matplotlib
pip install wfdb      # EHG (PhysioNet WFDB records)
pip install mne       # seizure EEG (EDF files)
```

The shared figure style lives in `nndr_style.py`. **Note:** the per-figure
scripts import it as `figure_style`, so place it on the path as `figure_style.py`
(e.g. `cp nndr_style.py figure_style.py`) or adjust the import line.

---

## 1. Uterine EHG (parturition) — strongest evidence

**Figures:** 9 (representative recording), 10 (peak-aligned), 11 (45-subject
dataset-level summary).

**Data source.** Icelandic 16-electrode Electrohysterogram Database, PhysioNet
(Alexandersson et al., *Scientific Data* 2 (2015) 150017,
doi:10.1038/sdata.2015.17; PhysioNet, ref. [23]/[25] in the paper). Public.
Format: WFDB `.hea`/`.dat` signal files (physical units) plus `.atr`
annotation files marking contractions.

**Selected recordings / channels.**
- Representative recording (Figs. 9–10): `ice045_p_1of4` (script default;
  `ice005_p_1of3` is provided as an alternative). Channel subset `EHG9`–`EHG12`.
- Dataset-level analysis (Fig. 11): **123 recordings from 45 subjects**, all
  available EHG channels per record.

**Preprocessing.**
- Read WFDB signals with `physical=True` (already calibrated). **Do not** apply
  the 131.068 factor used for raw `.mat` files.
- Convert mV → µV.
- Crop the initial contaminated segment (`INITIAL_CROP_SEC = 200 s`) and edge
  effects (`EDGE_CROP_SEC = 60 s`); despike; detrend.
- Bandpass filter to the uterine band **0.08–2.0 Hz**.
- Downsample to **10 Hz**.

**Moving window / fit.** 60 s windows, 5 s step. Ridge-regularized local VAR(1)
fit with trace-preserving isotropic shrinkage `α = 0.1` (early windows can be
noisy). Planes extracted independently by M2 and M3.

**Reference / event.** EHG-derived activity envelope `s_EHG(t)` — the
channel-averaged magnitude of the analytic (Hilbert) signal (Eq. 63). Peaks of
this envelope define the alignment times in Fig. 10. For the dataset-level
analysis, high-activity windows = top 20% of the smoothed envelope within each
recording; baseline = bottom 50% (Eq. 67). Effect sizes: per-recording Spearman
correlation (Eq. 69) and Cliff's delta; each subject summarized by its median.

**Scripts.**
- [`ehg_nonnormal_rolling.py`](reproducibility/ehg_nonnormal_rolling.py) → Figs. 9 & 10.
- [`ehg_final_dataset_level.py`](reproducibility/ehg_final_dataset_level.py) → Fig. 11.

---

## 2. Epileptic seizure EEG

**Figures:** 12 (representative seizure), 13 (patient-pooled cohort).

**Data source.** CHB-MIT Scalp EEG Database, PhysioNet (Children's Hospital
Boston; Goldberger et al., *Circulation* 101 (2000) e215, ref. [23]). Public,
long-term pediatric scalp EEG with expert seizure onset/offset annotations,
sampled at **256 Hz**. The full download is ~42 GB and was processed on the
Risks-X server. Format: EDF (`chbXX/chbXX_YY.edf`).

**Selected recordings.** Representative: `chb01/chb01_03.edf`, with the
annotated seizure at ~0–65 s. Cohort: all successfully analyzed seizure
recordings.

**Preprocessing (via `mne`).**
- Retain EEG channels; drop dummy/non-EEG channels; normalize channel names.
- Bandpass **0.5–40 Hz**; **60 Hz** notch; downsample to **32 Hz**.
- Align each seizure to its annotated onset, `τ = t − t_onset` (Eq. 70); plot
  against the onset-relative window endpoint `τ_end` (Eq. 71).

**Moving window / fit.** 40 s windows, 2 s step. Ridge-regularized fit with mild
isotropic shrinkage `α = 0.05`. Planes by M2 and M3.

**Reference / event.** EEG mean field `m_EEG(t)` (Eq. 72); for the cohort, its
smoothed absolute amplitude on the onset-aligned grid. Onset-aligned curves are
summarized within each patient before pooling, so patients with many seizures do
not dominate the cohort median.

**Scripts.**
- [`fig11_seizure_single_window_end_patched.py`](reproducibility/fig11_seizure_single_window_end_patched.py)
  → Fig. 12. (The script reads a pre-computed rolling-diagnostics CSV and the
  EDF; the CSV is produced by the same rolling analysis.)
- Cohort/pooled script (Fig. 13) — *to be added.*

---

## 3. Freezing of gait

**Figures:** 14 (representative event), 15 (onset vs. baseline across events).

**Data source.** Daphnet Freezing of Gait dataset — a public benchmark of
wearable accelerometer signals from Parkinson's patients during walking tasks
designed to elicit freezing episodes (context refs. [24]). Format: annotated
acceleration channels; this analysis uses ankle accelerometry.

**Selected recording / event.** Representative: `S01R01`, ankle event 03,
annotated FOG event 1236.0–1245.6 s. The event is split into an **onset
interval** (first 4 s, entering freezing) and a **sustained interval**
(remainder, strongly suppressed motion).

**State / signal.** Three ankle acceleration axes `x(t) = [a_x, a_y, a_z]`
(Eq. 73); displayed behavioral signal is the acceleration magnitude `‖a(t)‖`
(Eq. 74).

**Moving window / fit.** 3 s windows, 0.5 s step. Ridge-regularized fit. Planes
by M2 and M3.

**Reference / event.** Annotated freezing onset. Dataset-level comparison uses
the onset interval vs. matched baseline windows taken outside the freezing
episode (a second post-episode rise in `R` is read as gait re-initiation, not
onset, and is excluded from the onset-vs-baseline comparison).

**Scripts.**
- [`fig13_fog_representative.py`](reproducibility/fig13_fog_representative.py) → Fig. 14.
  Replots from two exported CSVs
  (`..._plotted_acceleration_native.csv`, `..._plotted_metrics_rolling.csv`); no
  recomputation.
- Dataset-level onset-vs-baseline script (Fig. 15) — *to be added.*

---

## 4. Rhythmic push-up motion (included in this repo)

**Figure:** 16.

**Data source.** Original wearable inertial recording — Apple Watch, Core Motion
**user acceleration** (gravity already removed, units of *g* ≈ 9.8 m s⁻²) —
during rhythmic push-ups performed on two small unstable, deformable elastic
fitballs (Logic Workout protocol; https://logicworkoutapp.com; refs. [30,31]).
No externally annotated event; higher- vs. lower-amplitude samples are separated
from the recording's own fluctuation envelope.

**Data file.** [`data/pushup/raw_samples_Pushup_duo_27Dec2025..csv`](data/pushup/)
with columns:

| column      | meaning                              |
|-------------|--------------------------------------|
| `timestamp` | sample time (used to infer sampling interval) |
| `accX`      | Core Motion user acceleration, x (g) |
| `accY`      | Core Motion user acceleration, y (g) |
| `accZ`      | Core Motion user acceleration, z (g) |

See [`data/pushup/README.md`](data/pushup/README.md) for the exact schema.

**Preprocessing.** Subtract the residual per-channel mean; form the acceleration
magnitude (Eq. 74). High-acceleration class = **upper 15%** of the amplitude
envelope; low-acceleration class = remainder (the 15% cutoff is varied from 30%
down to 9% as a robustness check).

**Fit.** A **separate** local one-step operator is fit per class (Eq. 75) by
ridge regression, relative ridge level `3 × 10⁻³`, isotropic shrinkage
`α = 0.04`. M2 and M3 give one `R` per class and method.

**Cycle-removal robustness check.** A phase-adaptive harmonic model (Eq. 81,
`K = 3` harmonics; instantaneous phase from the first PC of the 0.2–1.2 Hz-band
acceleration) removes the voluntary push-up rhythm; the non-cycle residual is
band-limited to **1–15 Hz** (main check; widened to 1–25 Hz as a further check)
and the analysis is repeated.

**Timing test.** Reaction coordinate `yᵣ(t)` (Eq. 76); its correlation with the
high-acceleration envelope is compared against a circular-shift null (1000
shifts, minimum shift 3 s) to get a standardized `z` and one-sided Monte-Carlo
`p` (Eqs. 77–80).

**Analysis.** The full push-up analysis code (raw + phase-adaptive cycle-removal
panels producing Fig. 16) is available from the authors on request. The steps
above, together with the data file in `data/pushup/`, fully specify the workflow.

> The synthetic-benchmark figures (Figs. 3–8) are reproduced directly from the
> packaged toolkit — see the *Reproducing the paper's synthetic benchmarks*
> section of the [README](README.md).
