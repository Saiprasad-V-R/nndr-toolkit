# Statistical controls

Functions support rebuttal R1.3, R2.8 and Section 6.2, but are not the full registered empirical protocol. All time series are `(channels, samples)`; use the same preprocessing and estimation on observed and surrogate data.

- `portmanteau_test(E, max_lag=10, fitted_df=0)`: centered residuals, Cholesky-whitened, adjusted multivariate Ljung–Box statistic `T² sum_l ||C_l||_F²/(T-l)`; `C_l = W[:,l:] @ W[:,:-l].T/T`. Asymptotic chi-square reference with `d²h-fitted_df` degrees of freedom; no correction for data-adaptive plane extraction. Do not interpret default p-values as calibrated for fitted NNDR residuals. Singular residual covariance is rejected.
- `squared_residual_test(X, method='M3', max_lag=10, n_surrogates=199, lam_rel=.003, shrinkage=0, seed=0)`: sum of squared autocorrelations of squared projected residuals. Refit the operator and plane on each multivariate shared-Fourier-phase surrogate. Preserves cross-periodograms, not marginal amplitude distributions. This is phase-only, not multivariate IAAFT. Fixed M2 seed for consistent optimization. All nulls must yield finite diagnostics; fails explicitly otherwise.
- `variance_matched_surrogate(F,X,match_variance=True,seed=0)`: real Schur normal reference with identical eigenvalues, driven by centered residual covariance. Solve `S = Fn S Fn.T + C` and multiply both S and C by `trace(C0)/trace(S)` if matching variance. The stationary marginal variance is matched in expectation, not realized sample variance or each channel variance. Uses existing default Schur order, not decreasing-modulus ordered Schur as proposed in the rebuttal. Draw initial condition from stationary Gaussian distribution. No time-varying variance envelope. Return simulated X, normal map, covariance matrices and variance factor.
- `surrogate_test(observed,null,alternative='greater')`: one-sided rank test `(1 + count(null >= observed))/(B+1)` (reverse for less); percentile uses midranks. Monte Carlo p-value requires exchangeability, which is not automatic for fitted nulls. No nonfinite results are dropped.
- `block_bootstrap(n_samples,block_length,n_resamples=199,seed=0)`: circular moving-block indices. Apply synchronously across channels or to aligned observed predictor-response pairs, never create artificial transitions at block boundaries. Choose block length before inference. No automatic selection.
- `bootstrap_confidence_interval(observed,replicates,confidence=.95,method='percentile')`: percentile or bias-corrected (`method='bc'`, not BCa), includes bias and standard error. Coverage requires calibration; biased/skewed estimators and temporal dependence need suitable resampling.
- `unwrap_rates(G,reference,dt=1)`: two-dimensional real log nearest a trusted continuous reference in Frobenius norm. Complex pair branch chosen analytically; distinct positive real pair supported. Repeated/negative real spectrum rejected; ties flagged. Does not unwrap general N-dimensional maps and is not wired into `markov_rate_consistency`.

## Example

```python
import nndr
P,Y=nndr.adjacent_pairs(X)
result=nndr.analyze(X,method='M3',dt=dt,lam_rel=.003)
E=result.Q.T @ (Y-result.Fhat@P)
white=nndr.portmanteau_test(E,max_lag=5)
null=nndr.variance_matched_surrogate(result.Fhat,X,seed=1)
nonlinear=nndr.squared_residual_test(X,max_lag=5,n_surrogates=199,seed=1)
```

## Remaining empirical calibration

No adequacy thresholds are selected here. Calibrate on synthetic and baseline windows, not event outcomes. Full multivariate amplitude-adjusted Fourier surrogates, modulus-ordered Schur basis, time-varying variance nulls, general ambient branch matching, bootstrap coverage, and dataset-level placebo/cohort tests remain separate tasks. Report failed/admissible fractions; do not drop inconvenient nulls. SciPy is required for statistical inference and Lyapunov solutions.
