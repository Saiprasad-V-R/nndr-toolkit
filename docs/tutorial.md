# Tutorial: from a known system to observations

[Home](../README.md) · [Quick start](quickstart.md) · [API reference](api_reference.md)

Run the self-contained example from the repository root after installing `.[dev]`:

```bash
python examples/tutorial_nndr.py --out results/tutorial
```

The script generates a stable oscillatory two-dimensional block, embeds it in 12 dimensions, simulates 5,000 samples, fits one common map, compares M2/M3 with the planted plane, evaluates held-out prediction and residual checks, constructs a normal reference, and saves a rolling-window CSV and plot. It uses fixed seeds. This is an illustrative stochastic model, not a calibrated recovery guarantee. No empirical data or internet connection is needed.

## 1. Known continuous or discrete dynamics

```python
import numpy as np
import nndr
from scipy.linalg import expm

L = nndr.real_generator(1., 2., 1.5)
dt = .1
F = expm(dt * L)
r = nndr.analyze_operator(F, method='M3', dt=dt)
print(r.R_A, r.R_sigma, r.diagnostics['stable_reduced'])
print(nndr.optimal_gain(L, .5, continuous=True))
```

Pass a discrete map to `analyze_operator`. For real spectra use `Delta_rate`; for complex spectra use `beta`. Use `R_A` and `R_sigma` explicitly rather than combining them into a single generic threshold.

## 2. Infer from a multivariate record

```python
L = nndr.complex_generator(1., 2., 1.5)
F, Q_true = nndr.embedded_map(L, dimension=12, dt=dt, seed=1)
X = nndr.simulate_trajectory(F, samples=5000, seed=2)
Fhat, info = nndr.fit_relative_operator(X, lam_rel=.003,
                                      shrinkage=.04, return_info=True)
a = nndr.analyze_operator(Fhat, method='M2', dt=dt)
b = nndr.analyze_operator(Fhat, method='M3', dt=dt)
print(info, nndr.principal_angle(a.Q, b.Q))
print(nndr.principal_angle(b.Q, Q_true))
print(nndr.plane_correlation(X, b.Q, nndr.mean_field(X)))
```

In experimental data the true plane is unavailable. Report agreement, closure error and model adequacy rather than claiming recovery error or a causal plane. The two methods share the same estimate, so their agreement is not independent validation of the fit.

## 3. Use your own CSV

Assume the CSV has `time`, `sensor1`, `sensor2`, `sensor3` columns; replace with the actual column names and units. The time column must be numeric seconds and regularly sampled.

```python
import pandas as pd
D = pd.read_csv('recording.csv')
t = D['time'].to_numpy(dtype=float)
X = D[['sensor1', 'sensor2', 'sensor3']].to_numpy(dtype=float).T
if len(t) < 3 or not np.isfinite(t).all() or not np.isfinite(X).all():
    raise ValueError('Require finite observations and timestamps')
d = np.diff(t)
if np.any(d <= 0) or not np.allclose(d, d[0], rtol=1e-5, atol=1e-9):
    raise ValueError('Require a justified regularly sampled grid')
dt = float(d[0])
```

Use justified filtering and anti-aliasing before downsampling. Do not silently convert missing data to zero or independently scale channels. For a local fluctuation analysis explicitly center the window. This is a descriptive offline analysis, not a forecasting protocol.

```python
window, step = round(40/dt), round(2/dt)  # illustrative seconds
if window < max(3, X.shape[0]) or window > X.shape[1] or step < 1:
    raise ValueError('Adjust window/step to data and sampling')
rows = []
for start in range(0, X.shape[1]-window+1, step):
    end = start + window
    Z = X[:, start:end].copy()
    Z -= Z.mean(axis=1, keepdims=True)
    Fhat = nndr.fit_operator(Z, lam_rel=.003, shrinkage=.04)
    for method in ('M2', 'M3'):
        r = nndr.analyze_operator(Fhat, method=method, dt=dt)
        rows.append(dict(time=t[0]+end*dt, method=method,
                         RA=r.R_A, Rsigma=r.R_sigma, support=r.support,
                         stable_full=r.diagnostics['stable_full'],
                         stable_reduced=r.diagnostics['stable_reduced'],
                         rate_status=r.diagnostics['rate_status']))
pd.DataFrame(rows).to_csv('window_results.csv', index=False)
```

This explicit loop fits once per window; it uses the exclusive window endpoint. It does not suppress failed fits: inspect exceptions and, for batch processing, explicitly log failed windows rather than silently impute them. The 40/2 s choice is illustrative and not universal. For windows that need no local centering, `rolling_analyze` is the simpler interface. Retain NaNs in both cases.

## 4. Assess the model and compare with a normal reference

For a centered, regularly sampled record `X` and its fitted map, use consistent estimation choices. In a chronological held-out analysis, avoid preprocessing that estimates parameters from the test segment. The synthetic example needs no centering for its zero-mean generating model.

```python
Fhat = nndr.fit_operator(X, lam_rel=.003, shrinkage=.04)
r = nndr.analyze_operator(Fhat, method='M3', dt=dt)
a = nndr.model_adequacy(X, Fhat, r.Q, dt=dt, lam_rel=.003, shrinkage=.04)
print(a['prediction'], a['lag_errors'], a['markov']['status'])
P, Y = nndr.adjacent_pairs(X)
E = r.Q.T @ (Y - Fhat @ P)
print(nndr.portmanteau_test(E, max_lag=5))
if r.diagnostics['stable_full']:
    null = nndr.variance_matched_surrogate(Fhat, X, seed=10)
    print(nndr.spectral_radius(null['F']), nndr.sigma_max(null['F']))
```

Model adequacy assumes enough samples and nonsingular covariances for the relevant diagnostics. Its output is descriptive; invalid lag logarithms are flagged. Whiteness p-values are unadjusted for data-dependent plane selection. A normal null must be refitted through the same pipeline for a comparison of inferred statistics. To explore the phase-only squared-residual test on the synthetic example, use `nndr.squared_residual_test(X, method='M3', max_lag=5, n_surrogates=99, lam_rel=.003, shrinkage=.04, seed=10)`. Surrogate replication count and all inference thresholds require scientific justification. For circular block bootstrap, apply indices synchronously to observations or aligned predictor-response pairs; never fit across artificial block joins. See [statistical controls](statistical_controls.md).
