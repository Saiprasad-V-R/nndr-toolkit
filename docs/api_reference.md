# Public API reference

[Documentation](index.md) · [Quick start](quickstart.md) · [Tutorial](tutorial.md) · [Mathematics](mathematical_framework.md)

Import with `import nndr`. The following call forms describe the public API of NNDR 2.0.0. `X` is real, finite, channels by samples; `F` is a discrete square map; `L` is a continuous generator; `Q` is an orthonormal two-column plane; `Gamma` is a real 2-by-2 compression. `dt` is the sampling interval, not the window length. Keyword-only parameters are shown after `*`. No routine automatically centers or normalizes channels. Install `.[dev]` for all examples.

## Main inference interface

| Call | Returns / behavior |
|---|---|
| `fit_operator(X, *, lambdas=None, lam=None, lam_rel=None, frac_train=.8, seed=0, shrinkage=0., legacy_pairs=False)` | Fitted square map. Observed adjacent pairs by default. Absolute ridge or relative ridge; without a fixed penalty, select from an absolute grid by seeded random split. |
| `analyze(X, *, method='M2', lambdas=None, lam=None, lam_rel=None, frac_train=.8, seed=0, shrinkage=0., iters=250, dt=1., threshold='rates')` | `NNDRResult` with fitted operator and plane. |
| `analyze_operator(A, *, method='M2', dt=1., threshold='rates', **method_kwargs)` | `NNDRResult` from a supplied discrete map; `Fhat=None`. |
| `rolling_analyze(X, *, window, step=1, method='M2', lam=None, lam_rel=None, lambdas=None, shrinkage=0., seed=0, iters=250, skip_degenerate=True, dt=1., threshold='rates')` | List of dictionaries, one per complete window. `window`, `step` in samples. Currently `skip_degenerate` does not alter the loop or catch exceptions. |

Do not combine `lam_rel` with `lam` or `lambdas`. Use `threshold='legacy'` only to reproduce historical results; it also restores the historical zero predictor pair in `analyze`. `iters` applies to M2. The `seed` controls M2 initialization and the legacy ridge selection split; it is not the seed of the observed experiment.

### NNDRResult

| Field | Meaning |
|---|---|
| `R_A`, `R_sigma` | Corrected delayed variance-response and exact one-step ratios, respectively. NaN if inadmissible. |
| `R`, `Kc` | Selected ratio and its critical K; default is the rate-based criterion. |
| `Delta` | Real rate splitting or complex beta, depending on spectral family; legacy mode uses multiplier splitting. |
| `K` | Reduced non-normality index (infinite for defective repeated pair). |
| `support` | Normalized L1 reaction-direction support; NaN for flagged unidentifiable revised M3 direction. |
| `Q`, `r_hat`, `n_hat` | Basis and its columns; revised M3 orders reaction/input directions. |
| `method`, `ok` | Extraction key and selected spectral admissibility flag, not general model validity. |
| `Fhat` | Fitted full operator for `analyze`; None for `analyze_operator`. |
| `extra`, `diagnostics` | Method-specific dictionary and reduced/full-map diagnostics. |
| `summary()` | Human-readable one-line string. |

The rolling output contains `t_start`, `t_end` (exclusive), `t_center`, `R`, `Delta`, `K`, `support`, `plane_identifiable`, `direction_identifiable`, `ok`, `R_A`, `R_sigma`, `Kc_A`, `Kc_sigma`, `Delta_rate`, `beta`, `rhoF`, `sigmaF`, `rhoGamma`, `sigmaGamma`, `dF`, `spectrum_type`, `rate_admissible`, `rate_status`, `stable_full`, `stable_reduced`, `stable_amplifying`, `dt`, `threshold`. It does not return the full fitted matrices. Flags for M1/M2 may be None. Use explicit per-window fitting when matrices, centering, or per-window exception handling are needed.

## Plane geometry and observables

| Call | Meaning / output |
|---|---|
| `method1(F, *, dt=1., **kwargs)` | Eigenbasis-SVD baseline dictionary; `eps` passed to historical method. |
| `method2(F, *, dt=1., **kwargs)` | Alternating directed coupling, `iters=250, seed=0` forwarded. No global certificate. |
| `method3(F, *, tol=1e-12, dt=1.)` | Commutator plane with directed ordering. |
| `order_response_plane(F, Q, *, tol=1e-12)` | Tuple (ordered Q, coupling, direction-identifiable flag). |
| `project_operator(F, Q)` | `Q.T @ F @ Q`. |
| `reaction_support(r)` | Normalized L1 support. Zero vector gives NaN. |
| `mean_field(X)` | Arithmetic mean over channels, length T. |
| `response_coordinate(X, r)`, `input_coordinate(X, n)` | Directional projections, requiring unit vectors. |
| `plane_correlation(X, Q, observable=None, *, return_direction=False)` | Maximum absolute centered correlation with observable (default mean field); optional tuple with ambient unit direction. Constant observable or rank-zero coordinates yield NaN. |
| `principal_angle(Q, P)` | Largest principal angle, degrees. |
| `relative_operator_error(Fhat, F)` | Relative Frobenius error, undefined if reference norm zero. |
| `capture_fraction(F, Q)` | Reduced-to-full largest singular value ratio. |
| `commutator(F)`, `commutator_gap(F)` | Symmetric commutator and minimum separation from complementary eigenspace (infinity in 2D). |
| `closure_error(F, Q)` | Relative leakage of the plane under F. |

Method dictionaries include `Q`, `r`, `n`, `Gamma`, `diagnostics`, `okG` (revised spectral admissibility) and `legacy_okG`. M3 adds `Q_comm`, `coupling`, `commutator_gap`, `plane_identifiable`, `direction_identifiable`. Overall direction signs are arbitrary. Use `method3` rather than the historical `method3_commutator` for new directional analyses.

## Reduced and ambient spectral measures

| Call | Meaning |
|---|---|
| `threshold_diagnostics(Gamma, dt=1.)` | Both ratios and spectral status dictionary. |
| `eigenvalue_splitting(Gamma, dt=1.)` | Real decay-rate splitting, NaN for complex/inadmissible cases. |
| `complex_splitting(Gamma, dt=1.)` | Frequency/decay beta, NaN for real/inadmissible cases. |
| `non_normality_index(Gamma)` | K. |
| `eigenvector_conditioning(Gamma)` | Reduced eigenvector conditioning. |
| `critical_K_real(delta)`, `critical_K_complex(beta)` | Rate-based critical K for 0 <= delta < 1 or beta >= 0; zero gives infinity. |
| `variance_response_ratio(Gamma, dt=1.)`, `one_step_ratio(Gamma)` | R_A and R_sigma. |
| `spectral_radius(F)`, `sigma_max(F)` | Largest eigenvalue modulus and largest singular value. |
| `departure_from_normality(F)` | Henrici departure. |
| `condition_number(F)` | Operator 2-norm condition number, not eigenbasis conditioning. |
| `numerical_abscissa(L)` | Largest eigenvalue of symmetric part of continuous generator. |

`threshold_diagnostics` keys: `R_A`, `R_sigma`, `Kc_A`, `Kc_sigma`, `Delta_rate`, `beta`, `rate_splitting`, `rate_admissible`, `rate_status`, `stable_reduced`, `spectrum_type`, `K`, `dF`, `rhoGamma`, `sigmaGamma`, `sigmaGamma_min`, `stable_amplifying`, `rates`, `dt`. `compute_indices_from_Q(A,Q,*,dt=1.,threshold='rates')` additionally supplies `Gamma`, `trG`, `evG`, `rhoF`, `sigmaF`, `stable_full`, `threshold`, `Delta`, `KKc`, `Kc`, `okG`. Do not reconstruct a finite repeated-pair ratio from `K/Kc` when both are infinite. The kernel evaluates this limit directly.

## Estimation and dynamics

| Call | Meaning / output |
|---|---|
| `adjacent_pairs(X)` | Tuple P,Y with T-1 columns. |
| `relative_ridge(X, lam_rel)` | Absolute penalty from predictor energy per channel. Pass the predictor matrix, not the entire record. |
| `shrink_operator(F, alpha)` | Trace-preserving isotropic shrinkage, 0 <= alpha < 1. |
| `fit_relative_operator(X, *, lam_rel=.003, shrinkage=0., return_info=False)` | F, or (F, info); info keys `lam`, `lam_rel`, `condition`, `pairs`. |
| `ridge_fit_F(X, Y, lam)` | Low-level ridge fit on supplied predictor and response columns. |
| `select_lambda_ridge(X, Y, lambdas, frac_train=.8, seed=0)` | Selected absolute penalty and grid validation errors. Random pair split, not chronological forecast evaluation. |
| `optimal_gain(A, t=1, *, continuous=False)` | Norm gain at specified horizon, not maximum over all time. |
| `variance_response(A, t=1, *, continuous=False)` | Squared Frobenius norm of propagator. |
| `snapshot_gain(A, t=1, *, continuous=False)` | Dict: `gain`, `u`, `v`, `overlap`, `unique`. |
| `kreiss_constant(F, *, radii=None, n_angles=128)` | Finite-grid lower bound for discrete Kreiss constant. |
| `normal_surrogate(F)` | Real spectrum-matched normal map, default Schur ordering. |

For discrete dynamics `t` is an integer step count; for `continuous=True`, `A` must be a generator and `t` is physical time. See [estimation and adequacy](estimation_adequacy.md).

## Adequacy and uncertainty

| Call | Output / caveat |
|---|---|
| `prediction_skill(X, *, frac_train=.8, lam_rel=.003, shrinkage=0.)` | `skill`, `mse`, `baseline_mse`, `train_pairs`, `test_pairs`; chronological full vs diagonal VAR(1). |
| `lag_structure_error(F, X, lag)` | Relative uncentered lag covariance error. |
| `markov_rate_consistency(X, *, lags=(1,2,4), dt=1., imag_tol=1e-8)` | `relative_errors`, `status`, `rates`; no automatic log branch matching. |
| `residual_autocorrelation(E, *, max_lag=10, squared=False)` | Array (coordinates, max_lag), NaN for constant coordinates. |
| `phase_randomize(X, *, seed=0)` | Shared-phase Fourier surrogate preserving cross-periodograms. |
| `model_adequacy(X,F,Q,*,dt=1.,lags=(2,4),max_lag=10,lam_rel=.003,shrinkage=0.,frac_train=.8)` | `prediction`, `lag_errors`, `markov`, `residual_acf`, `squared_residual_acf`, `closure_error`. No verdict. |
| `portmanteau_test(E, *, max_lag=10, fitted_df=0)` | Statistic, df, pvalue, samples and reference; unadjusted asymptotic reference. |
| `squared_residual_test(X, *, method='M3', max_lag=10, n_surrogates=199, lam_rel=.003, shrinkage=0., seed=0)` | Observed statistic, null array, rank p-value, percentile, count, alternative. |
| `variance_matched_surrogate(F,X,*,match_variance=True,seed=0)` | `X`, `F`, `innovation_covariance`, `stationary_covariance`, `variance_scale`, `target_variance`. Stable full map required. |
| `surrogate_test(observed,null,*,alternative='greater')` | `pvalue`, `percentile`, `n_null`, `alternative`. Exchangeability required. |
| `block_bootstrap(n_samples,block_length,*,n_resamples=199,seed=0)` | Circular block indices, shape (replicates, samples). |
| `bootstrap_confidence_interval(observed,replicates,*,confidence=.95,method='percentile')` | `lower`, `upper`, `bias`, `se`, `confidence`, `method`, `z0`, `quantiles`; `bc` is bias-corrected, not BCa. |
| `unwrap_rates(G,reference,*,dt=1.,tol=1e-10)` | `rates`, `branch`, `ambiguous`, `error`; 2D only, reference-guided. |

See [statistical controls](statistical_controls.md) for assumptions and limitations. Neither a high ratio nor a nonsignificant residual test establishes causal dynamics.

## Synthetic systems and compatibility

| Call | Output |
|---|---|
| `real_generator(gamma1,gamma2,K)` | Stable real-spectrum 2D continuous generator, 0 < gamma1 < gamma2, K >= 0. |
| `complex_generator(gamma,omega,K)` | Stable oscillatory 2D continuous generator, gamma, omega > 0, K >= 0. |
| `embedded_map(L,*,dimension=12,dt=.1,background_rates=None,seed=0)` | (F, true Q), stable orthogonal embedding, unaliased complex frequency required. |
| `simulate_trajectory(F,*,samples,noise_std=.1,burn_in=500,seed=0)` | Simulated real N-by-T stochastic trajectory. |
| `validate_time_series(X,*,name='X',min_samples=3)` | Float array, finite 2D, N <= T. |
| `validate_operator(A,*,name='A')` | Finite square float array. |
| `validate_window(window,T,*,step=1)` | None or raises ValueError. |

Historical exports `method1_eigenbasis_svd(A,eps=1e-15)`, `method2_optimization(A,iters=250,seed=0)`, `method3_commutator(A,tol=1e-12)`, `METHODS`, `Kc_from_Delta(Delta)`, `twoD_metrics_from_Gamma(Gamma)` are retained for earlier workflows. The last returns (multiplier splitting, conditioning, K); its old threshold is not the default revised interpretation. `__version__` is the module version string; matches the distribution version; record the commit for reproducibility. The old `benchmark`, `observables`, `plotting` helpers remain available in their modules. Historical synthetic recovery keeps the old M3 raw support column and historical figures are not retroactively changed.
