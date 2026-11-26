# Estimation, response and model adequacy

Data are real, regularly sampled, channels by samples. No automatic centering, scaling, stabilization, or interpolation is performed. Centering and filtering are application choices. `F` denotes a discrete map; `L` denotes a continuous generator.

## Estimation (rebuttal R2.2, R2.3)

`adjacent_pairs(X)` gives `P=X[:,:-1]`, `Y=X[:,1:]`. The revised default of `fit_operator`, `analyze`, and `rolling_analyze` uses these observed pairs. `fit_operator(X,legacy_pairs=True)` and `analyze(...,threshold='legacy')` retain historical zero-to-first-sample pairing. This matters for the absolute-ridge random validation split even though the added zero predictor contributes nothing to the full-fit normal equations at fixed ridge.

`relative_ridge(P,lam_rel)` evaluates nu = lam_rel * trace(P P.T)/N. `fit_relative_operator(X,lam_rel=.003,shrinkage=0,return_info=True)` returns a fitted operator and the actual penalty, Gram condition number, and pair count. `fit_operator(X,lam_rel=...)` and the high-level APIs accept the same relative parameter; do not combine it with an absolute penalty or grid. If none is provided, the historical absolute-grid random split is retained (not a time-ordered out-of-sample assessment). `shrink_operator(F,alpha)` shrinks toward trace(F)/N times the identity. No stability rescaling. The value .003 is an example, not a universal setting.

## Dynamical response (R1.1, R2.4, Appendices B/C)

- `optimal_gain(F,t)` is ||F^t||_2; t must be an integer >=0. `optimal_gain(L,t,continuous=True)` uses exp(L*t). The gain is not squared and is at a specified horizon, not a maximum over time.
- `snapshot_gain(F,t)` returns gain, optimal input v, normalized output u, absolute overlap, and a numerical uniqueness flag.
- `variance_response(F,t)` is ||F^t||_F^2, the trace of an impulse isotropic covariance response; `continuous=True` uses the generator. Divide by N if a per-channel response is desired. This is not the cumulative covariance of a continuously driven process.
- `kreiss_constant(F,radii=None,n_angles=128)` is a finite grid lower bound for the discrete Kreiss constant, including its limiting value 1. Increase the grid for sensitivity. Not a certified supremum. Unstable maps return infinity; marginal maps are not supported.
- `normal_surrogate(F)` uses a real Schur basis and replaces each pair block by a normal rotation-dilation block, deleting off-block coupling. Spectrum is preserved. Default Schur ordering is not the decreasing-modulus ordering proposed in the rebuttal. Innovation covariance, variance matching and Monte Carlo calibration are separate tasks.

## Descriptive adequacy (R1.3)

- `prediction_skill(X,frac_train=.8,lam_rel=.003,shrinkage=0)` measures 1 - held-out full-model MSE/diagonal-model MSE (baseline is diagonal VAR1). Both models use identical chronological training pairs, ridge and shrinkage convention; no event labels are used. A negative value indicates worse prediction than the diagonal baseline.
- `lag_structure_error(F,X,lag)` evaluates ||C_lag-F^lag C0||_F/||C_lag||_F using matching uncentered lagged pairs. Use centered data when desired. `lag` is a positive integer number of samples.
- `markov_rate_consistency(X,lags=(1,2,4),dt=1)` fits unregularized lag maps and compares principal logarithms divided by physical lag. Singular covariance and inadmissible logarithms are flagged, not repaired; no branch unwrapping is implemented. Interpretation requires sampling sufficiently fine to exclude aliasing. Finite sample errors are expected.
- `residual_autocorrelation(E,max_lag=10,squared=False)` provides per-coordinate centered ACFs; constant coordinates return NaN. `phase_randomize(X,seed=0)` uses shared random Fourier phases, preserving the multivariate cross-periodogram but not marginal amplitude distributions. It is a building block, not a complete calibrated surrogate test.
- `closure_error(F,Q)` measures relative leakage out of the plane, ||FQ-Q(Q.T FQ)||_F/||FQ||_F. It is an additional diagnostic, not a substitute for the rebuttal's tests.
- `model_adequacy(X,F,Q,...)` aggregates raw diagnostics. Its residual ACFs are projected into Q; held-out skill refits on training only. It does not issue a statistical verdict or silently exclude windows. Use consistent estimation settings in both fitting and assessment.

The [statistical API](statistical_controls.md) now provides a plane-residual portmanteau statistic, a phase-only squared-residual surrogate test, stationary variance/residual-covariance matched simulation, and reference-guided 2D logarithm branch selection. The branch helper is not wired into ambient lag consistency. Automated lag-gain comparison, response-delay-based lag selection, modulus-ordered Schur controls, and baseline-only cutoff calibration remain outside the current implementation. The rebuttal contains placeholders for numerical thresholds, so no empirical cutoff is asserted here. SciPy is needed for matrix logarithms, exponentials and Schur decomposition (install `.[dev]` or `.[synthetic]`).

The prior synthetic figure scripts are retained unchanged: their raw M3 support column uses the historical basis and must not be interpreted as corrected reaction support. Plane-invariant ratios are unaffected. The legacy benchmark correlation helper is also unchanged.
