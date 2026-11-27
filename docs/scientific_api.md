# Scientific API: geometry and observables

All data matrices are real, with channels in rows and samples in columns. `F` is a discrete one-step map; `L` is a continuous-time generator. `Q` is an orthonormal N-by-2 basis. `Gamma = Q.T @ F @ Q`. Angles are in degrees. A sample interval `dt` is positive; rates are inverse time units. No automatic stability rescaling is performed.

```python
import nndr
L = nndr.complex_generator(1.0, 2.0, 1.5)
F, Q_true = nndr.embedded_map(L, dimension=12, dt=0.1, seed=1)
X = nndr.simulate_trajectory(F, samples=5000, seed=2)
a = nndr.method3(F, dt=0.1)
G = nndr.project_operator(F, a['Q'])
print(nndr.variance_response_ratio(G, dt=0.1))
print(nndr.one_step_ratio(G))
print(nndr.reaction_support(a['r']))
print(nndr.plane_correlation(X, a['Q'], nndr.mean_field(X)))
```

## Plane extraction and geometry

- `method1(F, dt=1, **kwargs)` and `method2(F, dt=1, **kwargs)` wrap the historical algorithms. M2 is alternating optimization, not a certified global solution.
- `method3(F, tol=1e-12, dt=1)` extracts the two extreme commutator eigenvectors, then solves the two-dimensional directed coupling problem exactly (manuscript Eqs. 37–42). Its `Q` has columns `[r,n]`; `Q_comm` is the original raw plane. `plane_identifiable` and `direction_identifiable` are numerical flags, not statistical confidence measures. A tied or rotationally degenerate direction is flagged. Overall sign is arbitrary.
- The public method dictionaries contain `diagnostics` (both revised thresholds) and corrected `okG` (rate admissibility). `legacy_okG` preserves the historical flag. `method3_commutator` is retained unchanged for older scripts; it returns the raw commutator basis, not the ordered pair.
- `analyze_operator`, `analyze`, and `rolling_analyze` now use the ordered M3 basis by default. With `threshold='legacy'`, the old basis and threshold are used. For revised M3, support is NaN when the direction is flagged unidentifiable; `ok` is still spectral admissibility, separate from direction identifiability. Rolling results include both new flags; other methods have no corresponding flag yet.
- `order_response_plane(F,Q)` returns ordered Q, maximum coupling, and directional identifiability flag inside the supplied plane. Write `v=(cos(theta),sin(theta))`, `u=(-sin(theta),cos(theta))` in the plane. For `Gamma=[[a,b],[c,d]]`, the signed transverse coupling is `(c-b)/2 + (c+b)/2*cos(2*theta) + (d-a)/2*sin(2*theta)`. Maximizing its absolute value and selecting the sign of `u` yields the exact optimum.
- `project_operator(F,Q)`, `principal_angle(Q,P)` (Eq. 23), `relative_operator_error(Fhat,F)` (Eq. 22), `capture_fraction(F,Q)`, `commutator(F)`, `commutator_gap(F)` (minimum separation from the complementary eigenspace; infinity for N=2), `reaction_support(r)` (Eq. 61, NaN for zero direction).

## Spectral diagnostics

- `eigenvalue_splitting(Gamma,dt=1)` and `complex_splitting(Gamma,dt=1)` return real decay-rate splitting and beta, respectively. An inadmissible or wrong spectral family returns NaN.
- `non_normality_index(Gamma)`, `eigenvector_conditioning(Gamma)`, `critical_K_real(delta)`, `critical_K_complex(beta)`, `variance_response_ratio(Gamma,dt=1)`, `one_step_ratio(Gamma)` delegate to the revised formulas or threshold kernel. At zero splitting, the critical K limit is infinity; a repeated nonnormal pair has infinite K but finite R_A computed without dividing inf by inf. At a scalar repeated pair K=0 and the reported conditioning 1 describes an orthonormal choice, not a unique eigenbasis.
- `spectral_radius(F)`, `sigma_max(F)`, `departure_from_normality(F)`, `condition_number(F)` (operator 2-norm condition number), `numerical_abscissa(L)` (continuous generator). General-dimensional Henrici departure uses norm subtraction and can lose relative precision near normality.
- R_A requires stable reduced principal-logarithm admissibility. R_sigma requires Schur stability only. These two diagnostics test distinct phenomena. Inadmissible ratios are NaN, not zero.

## Observables

`mean_field(X)` (Eq. 24), `response_coordinate(X,r)` and `input_coordinate(X,n)` compute observable coordinates. `plane_correlation(X,Q,observable=None,return_direction=False)` computes the exact Eq. (25) maximum using the least-squares projection of a centered observable onto centered plane coordinates. If requested it also returns a unit ambient optimal direction. The default observable is the mean field. A constant observable or rank-zero plane has undefined correlation (NaN); rank-deficient nonzero planes are supported. Zero projection has score zero but no unique returned direction. The old `best_plane_corr` remains for historical compatibility; it uses a covariance heuristic rather than the exact maximizer.

## Compatibility and remaining work

Stage 2 synthetic recovery scripts deliberately use the historical low-level extraction functions so the already validated plane-invariant figures remain reproducible. Their M3 support column describes the first raw basis vector, not the corrected reaction direction; use the ordered API for new directional analysis. The legacy benchmark helper also retains the old observable-correlation heuristic. These historical scripts are intentionally frozen. Use the revised public functions for new results. Relative ridge, adjacent-pair fitting, model adequacy, time-dependent gain and Kreiss calculations are now available; see [estimation and adequacy](estimation_adequacy.md), [statistical controls](statistical_controls.md), and the [public API](api_reference.md). The validated representative EEG example is in [reproducibility](../manuscript/README.md).
