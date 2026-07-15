# Reproducibility

This document records, for each empirical dataset in the paper *Inferring
Non-Normal Amplification Geometry from Multivariate Time Series* (Saiprasad,
Troude & Sornette, 2026), the exact data source, the selected recordings and
channels, the preprocessing, the moving-window parameters, the analysis steps,
and the figure-generation workflow.

## Scope and design

This repository is released as a **reusable calibration and analysis toolkit**
(the `nndr` package). Three of the four empirical datasets are **public**
(Icelandic EHG, CHB-MIT seizure EEG, Daphnet freezing of gait); a direct link is
given in each section below. The fourth (push-up inertial recording) is an
original recording and is **included in this repository** under
[`data/pushup/`](data/pushup/).

**Public dataset links**

- Icelandic 16-electrode EHG (PhysioNet): https://physionet.org/content/ehgdb/1.0.0/
- CHB-MIT Scalp EEG (PhysioNet): https://physionet.org/content/chbmit/1.0.0/#files-panel
- Daphnet Freezing of Gait (UCI): https://archive.ics.uci.edu/dataset/245/daphnet+freezing+of+gait

All four empirical analyses use the common moving-window construction of
Section 6.2 of the paper: fit a local one-step operator `F̂ⱼ` by
ridge-regularized regression inside each window, apply optional trace-preserving
isotropic shrinkage `α`, extract a two-dimensional response plane by two
independent methods (M2, optimization-based directed coupling; M3, commutator),
and compute the reduced diagnostics `R = K/Kc(Δ)`, the reaction support `S(r̂)`,
and the reduced eigenvalue splitting `Δ`. Every diagnostic is displayed at the
**window end time** (causal/trailing convention), so no sample later than
`t_end` contributes to the value shown at `t_end`.

The reduced diagnostics are computed exactly as in the packaged toolkit
(`nndr.reduce2d`, `nndr.core`); the explicit formulas are collected below so this
document is self-contained.

### Common construction and diagnostics (explicit forms)

Inside window `j` (samples `t_j … t_j + W − 1`), form the lagged data matrices

```
X = [ x_{tj}   x_{tj+1} … x_{tj+W-2} ]
Y = [ x_{tj+1} x_{tj+2} … x_{tj+W-1} ]
```

and fit the one-step operator by ridge regression, then stabilize it:

```
F̂_j        = Y Xᵀ ( X Xᵀ + λ I_N )⁻¹
F̂_{j,reg}  = (1 − α) F̂_j + α μ_j I_N ,   μ_j = tr(F̂_j) / N
```

Extract the orthonormal response plane `Q = [ r̂ , n̂ ]` (by M2 and M3) and
project the operator onto it:

```
Γ = Qᵀ F̂_{j,reg} Q          (2 × 2 reduced operator)
```

From the eigenvalues `λ₁, λ₂` and unit eigenvectors `p₁, p₂` of `Γ`:

```
Δ      = | (λ₁ − λ₂) / (λ₁ + λ₂) |                       (reduced eigenvalue splitting)
c      = | ⟨p₁, p₂⟩ |
κ2D    = sqrt( (1 + c) / (1 − c) )                       (reduced non-orthogonality)
K      = ( κ2D − 1/κ2D ) / 2                             (reduced non-normality index)
Kc(Δ)  = [ sqrt(1 − Δ²) / ( 1 − sqrt(1 − Δ²) ) ]^{1/2} , 0 ≤ Δ < 1   (reduced threshold)
R      = K / Kc(Δ)                                       (normalized reduced ratio)
```

Reaction-direction support (spread of `r̂` across channels):

```
S(r̂) = ‖r̂‖₁ / ( √N · ‖r̂‖₂ ) ,   0 < S(r̂) ≤ 1
```

`R = 1` is the reduced two-dimensional transient-amplification threshold, not a
clinical or behavioral boundary.

## Environment

```bash
python >= 3.9
pip install numpy scipy pandas matplotlib
pip install wfdb      # EHG (PhysioNet WFDB records)
pip install mne       # seizure EEG (EDF files)
```

---

## 1. Uterine EHG (parturition) — strongest evidence

**Figures:** 9 (representative recording), 10 (peak-aligned), 11 (45-subject
dataset-level summary).

**Data source.** Icelandic 16-electrode Electrohysterogram Database, PhysioNet
(Alexandersson et al., *Scientific Data* 2 (2015) 150017,
doi:10.1038/sdata.2015.17). Public — https://physionet.org/content/ehgdb/1.0.0/.
Format: WFDB `.hea`/`.dat` signal files (physical units) plus `.atr` annotation
files marking contractions.

**Selected recordings / channels.**
- Representative recording (Figs. 9–10): `ice012_p_1of4`. Channel subset
  `EHG9`–`EHG12`.
- Dataset-level analysis (Fig. 11): **123 recordings from 45 subjects**, all
  available EHG channels per record.

**Preprocessing.**
- Read WFDB signals with `physical=True` (already calibrated). **Do not** apply
  the 131.068 factor used for raw `.mat` files.
- Convert mV → µV.
- Bandpass filter to the uterine band **0.08–2.0 Hz**.
- Downsample to **10 Hz**.

**Moving window / fit.** 60 s windows, 5 s step. Ridge-regularized local VAR(1)
fit with trace-preserving isotropic shrinkage `α = 0.1` (early windows can be
noisy). Planes extracted independently by M2 and M3.

**Reference / event.** EHG-derived activity envelope — the channel-averaged
magnitude of the analytic (Hilbert) signal:

```
z_i(t)   = x_i(t) + i · H[x_i](t)          (H = Hilbert transform)
s_EHG(t) = (1/N) Σ_{i=1}^N | z_i(t) |
```

Peaks of `s_EHG(t)` define the alignment times in Fig. 10. For the dataset-level
analysis, high-activity and baseline windows are, per recording `j`,

```
H_j = { t_end : s_j(t_end) ≥ Q_0.80[ s_j ] }     (top 20%  — high-activity)
B_j = { t_end : s_j(t_end) ≤ Q_0.50[ s_j ] }     (bottom 50% — baseline)
```

where `Q_p[·]` is the p-quantile within recording `j`. Effect sizes are the
per-recording Spearman correlation and Cliff's delta,

```
ρ_j = ρ_Spearman( s_j(t_end), R_j(t_end) )
δ   = P(R_H > R_B) − P(R_H < R_B)
```

with each subject summarized by its median.

---

## 2. Epileptic seizure EEG

**Figures:** 12 (representative seizure), 13 (patient-pooled cohort).

**Data source.** CHB-MIT Scalp EEG Database, PhysioNet (Children's Hospital
Boston; Goldberger et al., *Circulation* 101 (2000) e215). Public —
https://physionet.org/content/chbmit/1.0.0/#files-panel. Long-term pediatric
scalp EEG with expert seizure onset/offset annotations, sampled at **256 Hz**.
The full download is ~42 GB and was processed on the Risks-X server. Format: EDF
(`chbXX/chbXX_YY.edf`).

**Selected recordings.** Representative: `chb01/chb01_03.edf`, with the
annotated seizure at ~0–65 s. Cohort: all successfully analyzed seizure
recordings.

**Preprocessing (via `mne`).**
- Retain EEG channels; drop dummy/non-EEG channels; normalize channel names.
- Bandpass **0.5–40 Hz**; **60 Hz** notch; downsample to **32 Hz**.
- Align each seizure to its annotated onset, and plot against the onset-relative
  window endpoint:

```
τ         = t − t_onset
τ_end,j   = t_end,j − t_onset
```

**Moving window / fit.** 40 s windows, 2 s step. Ridge-regularized fit with mild
isotropic shrinkage `α = 0.05`. Planes by M2 and M3.

**Reference / event.** EEG mean field

```
m_EEG(t) = (1/N) Σ_{i=1}^N x_i(t)
```

for the cohort, its smoothed absolute amplitude on the onset-aligned grid.
Onset-aligned curves are summarized within each patient before pooling, so
patients with many seizures do not dominate the cohort median.

---

## 3. Freezing of gait

**Figures:** 14 (representative event), 15 (onset vs. baseline across events).

**Data source.** Daphnet Freezing of Gait dataset — a public benchmark of
wearable accelerometer signals from Parkinson's patients during walking tasks
designed to elicit freezing episodes. Public —
https://archive.ics.uci.edu/dataset/245/daphnet+freezing+of+gait. This analysis
uses ankle accelerometry.

**Selected recording / event.** Representative: `S01R01`, ankle event 03,
annotated FOG event 1236.0–1245.6 s. The event is split into an **onset
interval** (first 4 s, entering freezing) and a **sustained interval**
(remainder, strongly suppressed motion).

**State / signal.** Three ankle acceleration axes and the acceleration
magnitude:

```
x(t)    = [ a_x(t), a_y(t), a_z(t) ]ᵀ
‖a(t)‖  = sqrt( a_x(t)² + a_y(t)² + a_z(t)² )
```

**Moving window / fit.** 3 s windows, 0.5 s step. Ridge-regularized fit. Planes
by M2 and M3.

**Reference / event.** Annotated freezing onset. Dataset-level comparison uses
the onset interval vs. matched baseline windows taken outside the freezing
episode (a second post-episode rise in `R` is read as gait re-initiation, not
onset, and is excluded from the onset-vs-baseline comparison).

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
magnitude `‖a(t)‖ = sqrt(a_x² + a_y² + a_z²)`. High-acceleration class = **upper
15%** of the amplitude envelope; low-acceleration class = remainder (the 15%
cutoff is varied from 30% down to 9% as a robustness check).

**Fit.** A **separate** local one-step operator is fit per class by ridge
regression, relative ridge level `3 × 10⁻³`, isotropic shrinkage `α = 0.04`:

```
x(t + Δt) ≈ F̂_q x(t) ,   q ∈ { low, high }
```

M2 and M3 give one `R` per class and method.

**Cycle-removal robustness check.** A phase-adaptive harmonic model removes the
voluntary push-up rhythm:

```
x_i(t) = Σ_{k=1}^K [ a_ik(t) cos(k φ(t)) + b_ik(t) sin(k φ(t)) ] + ε_i(t) ,   K = 3
```

where `φ(t)` is the instantaneous push-up phase from the first PC of the
0.2–1.2 Hz-band acceleration. The non-cycle residual `ε(t)` is band-limited to
**1–15 Hz** (main check; widened to 1–25 Hz as a further check) and the analysis
is repeated.

**Timing test.** With the reaction coordinate `y_r^m(t) = | (r̂^m)ᵀ x(t) |`, the
correlation between the high-acceleration envelope `e_high(t)` and `y_r^m(t)` is
compared against a circular-shift null (B = 1000 shifts, minimum shift 3 s):

```
c_high^m = corr( e_high(t), y_r^m(t) )
μ_null   = (1/B) Σ_b c_null,b
σ_null   = sqrt( (1/(B−1)) Σ_b ( c_null,b − μ_null )² )
z^m      = ( c_high^m − μ_null ) / σ_null
p^m      = ( 1 + #{ b : c_null,b ≥ c_high^m } ) / ( 1 + B )
```

so the smallest attainable `p` is `1 / (1 + B) = 9.99 × 10⁻⁴`.

**Analysis.** The full push-up analysis code (raw + phase-adaptive cycle-removal
panels producing Fig. 16) is available from the authors on request. The steps
above, together with the data file in `data/pushup/`, fully specify the workflow.

> The synthetic-benchmark figures (Figs. 3–8) are reproduced directly from the
> packaged toolkit — see the *Reproducing the paper's synthetic benchmarks*
> section of the [README](README.md).
