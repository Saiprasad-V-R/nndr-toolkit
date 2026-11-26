# NNDR threshold revision

For the fitted discrete-time map F and orthonormal plane Q, form Gamma = Q.T @ F @ Q. Version 2.0.0 also introduces observed adjacent-pair fitting, relative ridge regularization and directed M3 ordering. The [original snapshot](https://github.com/Saiprasad-V-R/nndr-toolkit/tree/v1.0.0) is preserved for historical comparison.

- `R_A` uses the real principal logarithm of Gamma divided by the sampling step, and the real-pair or complex-pair variance-response threshold (revised Appendices B.4 and C.4). Negative real or zero multipliers do not admit this principal logarithm. Such windows retain `R_sigma` if stable, but have `R_A=NaN`. The positive repeated-eigenvalue limit uses the divided difference of the logarithm; a defective block has infinite K but can have finite R_A and R_sigma.
- `R_sigma = dF / sqrt((1-|lambda1|^2)(1-|lambda2|^2))`, where dF is the Henrici departure. It is defined for stable reduced maps and exceeds one exactly when sigma_max(Gamma) exceeds one (C.7–C.9). No stability rescaling is performed. Full and reduced stability are reported separately. Neither ratio is a causal inference about physical dynamics.
- `R` (high-level) and `KKc` (low-level/benchmark) default to `R_A`. `Delta` means the rate splitting for real pairs or beta for complex pairs. `Delta_rate` and `beta` are separate unambiguous fields. `dt` is the sample interval, default one time unit; rates assume the principal logarithm branch and may alias. For a fixed Gamma the ratios are independent of the units assigned to dt; changing the sampled map may change the diagnostics.
- `threshold="legacy"` explicitly restores the original ratio, splitting and screening semantics; it may reproduce invalid historical values. Use only for reproducing the original publication. `twoD_metrics_from_Gamma`, `Kc_from_Delta` and `gamma_diagnostics` remain historical helpers. The revised ratios are separately present in legacy mode as well.
- The revised package is published as [2.0.0](https://github.com/Saiprasad-V-R/nndr-toolkit/releases/tag/2.0.0). The revised synthetic validations and representative EEG numerical parity audit have been completed. Scripts with independent historical formulas are not automatically migrated; the [reproducibility guide](reproducibility/README.md) specifies which analyses are verified.

```python
from nndr import analyze, rolling_analyze, threshold_diagnostics
res = analyze(X, method="M2", dt=1/256)
print(res.R_A, res.R_sigma, res.diagnostics["rate_status"])
records = rolling_analyze(X, window=400, step=20, dt=1/256)
old = analyze(X, method="M2", threshold="legacy")
```

Numerical kernel tested independently: 12 tests, including 3,000 random stable maps checked for one-step equivalence and orthogonal invariance. Near a threshold or a logarithm branch cut, floating-point classification remains sensitive; report numerical tolerance and sensitivity in empirical analyses rather than clipping the spectrum.
