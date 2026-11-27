# User guide: choose the workflow that matches your data

[Documentation](index.md) · [Quick start](quickstart.md) · [API reference](api_reference.md)

NNDR is designed for several distinct scientific tasks. Begin with the simplest appropriate workflow and add validation only as needed. The following examples use the current public API; install the `dev` extra for all dependencies.

## 1. I know the equations or a linear map

For continuous linear dynamics, sample the generator at a specified interval. For a nonlinear model, a Jacobian can describe local perturbations near a specified equilibrium, but not an arbitrary finite-amplitude trajectory.

```python
import numpy as np
import nndr
from scipy.linalg import expm

L = nndr.complex_generator(gamma=1.0, omega=2.0, K=1.5)
dt = 0.1
F = expm(L * dt)
r = nndr.analyze_operator(F, method='M3', dt=dt)
print(r.R_A, r.R_sigma, r.diagnostics['rhoGamma'])
```

For a two-dimensional known map, `threshold_diagnostics(F, dt=dt)` can compute the ratios directly; no plane search is required. Use `optimal_gain` to inspect a chosen time horizon rather than inferring an amplification magnitude from a threshold ratio. Always use a discrete map with `analyze_operator`.

## 2. I have high-dimensional simulation or sensor data

NNDR expects a real array of shape `(channels, samples)` with regular sampling. Each row represents one measured coordinate and each column one instant. Centering, detrending, filtering and channel scaling are explicit decisions, not automatic operations.

This reproducible synthetic record illustrates estimation when the equations are treated as unknown:

```python
L = nndr.complex_generator(1.0, 2.0, 1.5)
F, Q_true = nndr.embedded_map(L, dimension=12, dt=dt, seed=1)
X = nndr.simulate_trajectory(F, samples=5000, seed=2)
Fhat, info = nndr.fit_relative_operator(X, lam_rel=0.003,
                                      shrinkage=0.04, return_info=True)
a = nndr.analyze_operator(Fhat, method='M2', dt=dt)
b = nndr.analyze_operator(Fhat, method='M3', dt=dt)
print(info)
print(a.R_A, a.R_sigma, b.R_A, b.R_sigma)
print('Plane disagreement (degrees):', nndr.principal_angle(a.Q, b.Q))
print('M3 planted-plane error:', nndr.principal_angle(b.Q, Q_true))
```

Fit once so both extraction methods use the same map. With real measurements, replace `X` with your data and omit the planted-plane error. `lam_rel=0.003` and `shrinkage=0.04` are illustrative settings, not validated recommendations for every dataset. Assess sensitivity, conditioning and chronological held-out skill. For a single method, `nndr.analyze(X, ...)` combines fitting and extraction.

## 3. I want to know where amplification is concentrated

```python
if b.extra.get('direction_identifiable', True):
    print('Reaction support:', b.support)
    reaction = nndr.response_coordinate(X, b.r_hat)
    excitation = nndr.input_coordinate(X, b.n_hat)
    print('Mean-field alignment:', nndr.plane_correlation(X, b.Q))
    print('Closure error:', nndr.closure_error(Fhat, b.Q))
```

Plot the components of `b.r_hat` against channel labels to understand spatial participation. Use absolute components if the overall sign is immaterial, and do not equate a large coefficient with causal influence. `plane_correlation` measures alignment with the supplied observable (mean field by default), not predictive skill. Large closure error indicates that the full fitted dynamics leave the extracted plane substantially.

## 4. I want to follow dynamics that change over time

```python
rows = nndr.rolling_analyze(X, window=1000, step=250,
                            method='M3', dt=dt, lam_rel=0.003,
                            shrinkage=0.04)
end_time = np.array([row['t_end'] for row in rows]) * dt
ra = np.array([row['R_A'] for row in rows])
rs = np.array([row['R_sigma'] for row in rows])
```

The window and step are in samples. Choose them using time scales, sample size and local stationarity, not by maximizing an event-associated peak. `t_end` is the exclusive sample index; the time above describes the trailing window boundary, not its last observed sample. A change in an inferred statistic is temporally averaged by the window and can appear delayed. Overlapping estimates are dependent. Keep invalid windows as missing, not zero.

For local demeaning or explicit error logging, use the per-window loop in the [tutorial](tutorial.md). Do not treat the rolling `ok` flag as a test of overall model adequacy.

## 5. I have my own recording

For a numeric CSV, load its timestamp and measurement columns, validate regular sampling and transpose the channel table. The [CSV tutorial](tutorial.md#3-use-your-own-csv) provides a complete loading and windowed workflow. Use actual physical units and record the sampling interval; changing relative channel scales changes the Euclidean metric used by NNDR. For irregular timestamps, justify a regular grid before using this estimator. Fit preprocessing parameters on training data only for prospective prediction.

## 6. How should I report the result?

| Scientific question | Report |
|---|---|
| Is the full fitted map stable? | `stable_full`, `rhoF` (or `spectral_radius(Fhat)`) |
| Is the compression stable and the rate interpretation admissible? | `stable_reduced`, `rhoGamma`, `rate_admissible`, `rate_status` |
| Is there one-step norm amplification? | `R_sigma`, `sigmaGamma` |
| Is there an increasing interval of isotropic variance response? | `R_A` with rate admissibility |
| Where is the inferred reaction direction concentrated? | `r_hat`, `support`, direction identifiability where available |
| Is the inferred plane robust to extraction choice? | M2/M3 principal angle, closure, capture fraction |
| Is the model credible for this recording? | Held-out skill, residual/lag checks, sensitivity and suitable controls |

Do not infer physical causation or clinical prediction from these outputs. See [validation and uncertainty](validation_and_uncertainty.md) and [limitations](limitations.md).
