# Validation and uncertainty: when should I trust an NNDR result?

[Run the illustrated validation example](../examples/06_validation_and_uncertainty/README.md) for a comparison of independent and correlated innovations.

[Documentation](index.md) · [User guide](user_guide.md) · [Statistical controls](statistical_controls.md)

An amplification ratio is a deterministic diagnostic of an estimated reduced map. A large ratio alone is not evidence that the estimator captured the physical mechanism. Validation must address both the fitted map and the chosen plane.

## 1. Separate the questions

| Question | Diagnostic | Interpretation |
|---|---|---|
| Is the estimated map stable? | Full/reduced spectral radius | Necessary domain check for stable-amplification ratios. |
| Does the plane represent relevant dynamics? | Closure, capture fraction, M2/M3 angle | Descriptive checks; no universal acceptable cutoff. |
| Does the map predict held-out observations? | Full versus diagonal predictor skill | Negative skill means the full model is worse than this baseline. |
| Are lag dynamics consistent with a local Markov map? | Lag structure, principal-log rate consistency | Large discrepancies or branch ambiguity require scrutiny. |
| Are residuals consistent with the assumed model? | Residual correlations and portmanteau | Reference p-values are not adjusted for estimated plane selection. |
| Is the estimated statistic sensitive to noise or null dynamics? | Surrogates and block bootstrap | Match the control to the scientific hypothesis and refit the whole pipeline. |

## 2. A first model-adequacy check

The following block assumes a real, regularly sampled array `X` and the sampling interval `dt`, prepared as in the [user guide](user_guide.md). It uses identical estimator settings for the fitted map and chronological prediction check.

```python
import nndr

Fhat = nndr.fit_relative_operator(X, lam_rel=0.003, shrinkage=0.04)
r = nndr.analyze_operator(Fhat, method='M3', dt=dt)
a = nndr.model_adequacy(X, Fhat, r.Q, dt=dt,
                        lam_rel=0.003, shrinkage=0.04)
print(a['prediction'])
print(a['lag_errors'])
print(a['markov']['status'])
P, Y = nndr.adjacent_pairs(X)
E = r.Q.T @ (Y - Fhat @ P)
print(nndr.portmanteau_test(E, max_lag=5))
```

A nonsignificant residual test does not prove whiteness. A positive held-out score does not identify a causal map. Fit centering, scaling or other preprocessing on training data alone when assessing prospective prediction. Check validity of assumptions and covariance ranks before interpreting statistical quantities.

## 3. Compare with controls that answer a specific question

**Normal reference:** for a stable fitted map, `variance_matched_surrogate` replaces it by a normal map with the same spectrum and uses residual covariance to drive a simulated record. With variance matching enabled, the stationary *total* variance is matched in expectation. The resulting record need not have the same channel variances or time-varying envelope. Refit it with the same settings and re-extract the plane before comparing an inferred statistic. This is a stationary model-based reference, not a universal null.

**Shared-phase Fourier reference:** `phase_randomize` preserves the sample cross-periodogram; it does not enforce the original marginal distributions. `squared_residual_test` tests residual amplitude structure against such controls. The null choice must match the hypothesis.

**Uncertainty:** use `block_bootstrap` for circular moving-block indices and `bootstrap_confidence_interval` for percentile or bias-corrected intervals. Resample aligned predictor/response pairs synchronously; do not introduce spurious transitions across block boundaries. Neither block length nor coverage is guaranteed by the function. The Monte Carlo rank utility `surrogate_test` gives a finite-sample rank p-value under appropriate exchangeability; estimated plug-in nulls need not be exactly exchangeable.

## 4. When to withhold an interpretation

- A full or reduced map is unstable: the intended stable-amplification interpretation may not apply. Never force stability just to obtain a ratio.
- The principal logarithm is inadmissible: `R_A` is undefined, even if `R_sigma` remains available.
- The M3 plane or direction is unidentifiable: do not present its individual direction coefficients as unique.
- Fitted dynamics have poor predictive skill, strong residual structure, high leakage or marked sensitivity to preprocessing: investigate the mismatch rather than claim a mechanism.

Report failed or inadmissible windows and uncertainty, not just selected successful cases. No numerical screening cutoffs are supplied as universal standards. See [estimation and adequacy](estimation_adequacy.md), [statistical controls](statistical_controls.md), and [limitations](limitations.md) for exact implementation scope.
