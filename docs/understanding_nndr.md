# Understanding NNDR: from stability to amplification

[Documentation](index.md) · [Quick start](quickstart.md) · [Mathematics](mathematical_framework.md)

## 1. Why look beyond eigenvalues?

Consider a discrete dynamical system with state $x_k$ and update $x_{k+1}=Fx_k$. If every eigenvalue of $F$ lies inside the unit circle, the unforced linear system eventually decays to zero. This does not specify how large the response may become along the way.

Take two maps with exactly the same eigenvalues:

$$
F_{\mathrm{N}}=\begin{pmatrix}0.8&0\\0&0.5\end{pmatrix},\qquad
F_{\mathrm{NN}}=\begin{pmatrix}0.8&0.6\\0&0.5\end{pmatrix}.
$$

The second map transfers perturbations from its second coordinate into the first. Its eigenvectors are not orthogonal. Its strongest one-step response is therefore larger than that of the diagonal map, even though both decay asymptotically.

The numbers below are computed from the stated matrices, not fitted to data. The ratios use the revised definitions and, for the continuous-time criterion, the admissible principal logarithm with sampling interval one.

| Quantity | Normal map | Non-normal map |
|---|---:|---:|
| Spectral radius | 0.800 | 0.800 |
| Largest singular value | 0.800 | 1.051 |
| Optimal two-step norm gain | 0.640 | 1.028 |
| Variance-response ratio `R_A` | 0.000 | 0.812 |
| One-step ratio `R_sigma` | 0.000 | 1.155 |

Run the calculation yourself:

```python
import numpy as np
import nndr

for F in (np.diag([0.8, 0.5]), np.array([[0.8, 0.6], [0.0, 0.5]])):
    d = nndr.threshold_diagnostics(F, dt=1.0)
    print(nndr.spectral_radius(F), nndr.sigma_max(F))
    print(nndr.optimal_gain(F, 2), d['R_A'], d['R_sigma'])
```

This example is deliberately chosen so the two criteria disagree. One-step norm amplification can occur without a delayed increase in the isotropic impulse variance-response kernel. A single generic "amplification threshold" would obscure that distinction.

[Explore the computed four-panel figure and runnable example](../examples/01_why_non_normality_matters/README.md).

## 2. What is a response plane?

For many observed variables, studying every direction in the full state space is difficult. NNDR seeks two orthogonal unit vectors: an input direction $n$ and a reaction direction $r$. In an ordered plane, the basis $Q=[r,n]$ gives the compressed map

$$
\Gamma=Q^\top FQ.
$$

The plane records the directional interaction selected by the extraction method. The coupling $r^\top Fn$ measures how a unit input along $n$ transfers into $r$ in one step. It is not a causal connection between physical variables. The two directions need not be eigenvectors, and an extracted plane need not be invariant.

The components of $r$ indicate which measured coordinates participate in that direction. Signs depend on the arbitrary overall orientation of the basis, and changing units or independently scaling variables can change the inferred geometry. `reaction_support` summarizes how spread out the direction is, rather than identifying a causal subset of variables.

## 3. Three ways to extract a plane

| Method | Idea | How to use it |
|---|---|---|
| M1 | Eigenbasis-SVD baseline | Compare against a spectral construction. |
| M2 | Alternating optimization of directed coupling | Use when you want a coupling-based plane; inspect sensitivity to initialization. |
| M3 | Extreme eigenvectors of the symmetric commutator, followed by directed ordering | Use as a complementary geometric construction and inspect identifiability. |

A small angle between M2 and M3 suggests that the plane is insensitive to these two extraction rules. Both use the same estimated map, so this agreement cannot validate the estimation itself. In a normal or nearly degenerate case, a meaningful unique input/reaction direction may not exist.

## 4. Two different amplification diagnostics

For a stable reduced map, `R_sigma > 1` is exactly equivalent to its largest singular value exceeding one. It asks whether some unit perturbation grows in Euclidean norm over one sample interval.

For a stable reduced map admitting a real principal logarithm, `R_A > 1` indicates an interval of increasing isotropic impulse variance response in the associated continuous generator. This is not necessarily a peak above the initial variance. It is not a criterion for a generic observed variance time series.

For distinct real decay rates the rate splitting controls the critical non-normality. For complex rates, the relevant parameter is oscillation frequency divided by decay rate. The [mathematical framework](mathematical_framework.md) explains why these cases require separate formulas.

## 5. From a model to an experimental recording

If the linear dynamics are known, apply NNDR to a discrete map directly. For observations, first estimate a local predictor from adjacent samples, then extract the plane from that fitted map. Repeat in time windows if the dynamics evolve. A local linear predictor is not automatically the physical Jacobian, even when it is estimated from a nonlinear system.

Start with the [quick start](quickstart.md), then choose a workflow in the [user guide](user_guide.md). Use [validation and uncertainty](validation_and_uncertainty.md) before interpreting data-driven amplification as a robust scientific result.
