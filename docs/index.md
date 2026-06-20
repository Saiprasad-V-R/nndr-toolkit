# NNDR documentation

This page collects the conceptual background and a short API reference. For a
quick start, see the top-level [README](../README.md); for runnable code, see
[`examples/`](../examples/).

## Background

A linear system `x_{k+1} = F x_k` is *non-normal* when `F Fᵀ ≠ Fᵀ F`, i.e. its
eigenvectors are not orthogonal. Such a system can be asymptotically stable —
all eigenvalues inside the unit disk — yet still amplify perturbations strongly
over a finite time, because a perturbation can be routed through non-orthogonal
directions before it decays.

Eigenvalues alone do not reveal this. NNDR instead estimates the local operator
from data and isolates the two-dimensional plane responsible for the dominant
finite-time response.

## The reduced diagnostics

Given a fitted operator `F̂` and a response plane `Q = [r̂, n̂]`, project to the
reduced operator `Γ = Qᵀ F̂ Q` (a 2×2 matrix) and compute:

| Symbol | Meaning |
|--------|---------|
| `Δ` | reduced eigenvalue splitting, `|(λ₁−λ₂)/(λ₁+λ₂)|` |
| `K` | reduced non-normality index (0 for an orthogonal reduced eigenbasis) |
| `Kc(Δ)` | reduced transient-amplification threshold |
| `R = K / Kc(Δ)` | normalized reduced non-normality ratio |

The closed form of the threshold is

```
Kc(Δ) = sqrt( sqrt(1 − Δ²) / (1 − sqrt(1 − Δ²)) ),   0 ≤ Δ < 1.
```

`R` is a control parameter: `R < 1` is below the reduced threshold, `R = 1` is
the threshold, and `R > 1` is the amplification-prone regime. `R` is *not* a
phenomenon — it controls whether the reduced dynamics can transiently amplify.

## Choosing a method

- **M2 (optimization-based)** is the recommended default.
- **M3 (commutator-based)** is an independent construction; running both and
  comparing their planes (via `nndr.utils.principal_angle_deg`) is a useful
  internal consistency check, especially for empirical data with no ground truth.
- **M1 (eigenbasis-SVD)** is a transparent baseline; it is fragile when the
  fitted operator's eigenvectors are unstable, and is mainly useful for
  comparison.

## API reference

### High-level

- `analyze(X, method="M2", ...)` → `NNDRResult`
  Estimate the operator from `X` (shape `(N_channels, T_samples)`) and compute
  diagnostics.
- `rolling_analyze(X, window, step=1, method="M2", ...)` → `list[dict]`
  Sliding-window diagnostics for non-stationary data.
- `analyze_operator(A, method="M2", ...)` → `NNDRResult`
  Diagnostics from a known operator, skipping estimation.
- `fit_operator(X, lam=None, lambdas=None, shrinkage=0.0, ...)` → `np.ndarray`
  Ridge-fit the one-step operator only.

### `NNDRResult` fields

`R`, `Delta`, `K`, `Kc`, `support`, `Q`, `r_hat`, `n_hat`, `method`, `ok`,
`Fhat`, `extra`. The `summary()` method returns a one-line description.

### Lower-level

- `nndr.methods` — `method1_eigenbasis_svd`, `method2_optimization`,
  `method3_commutator`, and the `METHODS` registry.
- `nndr.reduce2d` — `compute_indices_from_Q`, `twoD_metrics_from_Gamma`,
  `Kc_from_Delta`.
- `nndr.ridge` — `ridge_fit_F`, `select_lambda_ridge`.
- `nndr.observables` — `best_plane_corr`.
- `nndr.utils` — `principal_angle_deg`, `spectral_radius`, and others.

### Benchmarking (optional extra)

- `nndr.benchmark` — `BenchConfig`, `run_benchmark`.
- `nndr.plotting` — `plot_median_vs_x`, `plot_threshold_misclass_vs_x`,
  `add_R_error_columns`, and others.

Install with `pip install "nndr-toolkit[benchmark]"`.
