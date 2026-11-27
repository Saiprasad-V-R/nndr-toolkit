# Synthetic validation: exact two-dimensional systems

This is the first synthetic reproduction experiment for the revised NNDR toolkit. It is a new validation of the corrected thresholds in revised Appendices B and C, not a numerical recreation of a numbered figure in arXiv v1.

## Run

From the repository root, in an activated Python environment:

```sh
python -m pip install -e '.[dev]'
python -m pip install scipy
python manuscript/synthetic/validate_thresholds.py --out results/synthetic/thresholds
```

On the previously validated macOS Python 3.9.6 environment, retain NumPy 1.26.4 and use `python -m pip install scipy==1.13.1`. The core generators themselves require only NumPy. SciPy is used by the independent numerical validation script, not by the NNDR threshold routine.

Outputs: `threshold_sweep.csv` (all cases and labels), `summary.json` (versions, tolerances and counts), `threshold_ground_truth.png`, and `threshold_ground_truth.svg`. No input data, random seed or network access is needed once dependencies are installed. `--no-figure` skips plotting; any disagreement in the sweep raises an error before plotting.

## Reusable generators

```python
from nndr.synthetic import real_generator, complex_generator
from nndr.thresholds import threshold_diagnostics
from scipy.linalg import expm

L = complex_generator(gamma=1., omega=2., K=1.5)
F = expm(0.1 * L)
print(threshold_diagnostics(F, dt=0.1))
```

`real_generator(gamma1, gamma2, K)` requires `0 < gamma1 < gamma2`; `complex_generator(gamma, omega, K)` requires positive gamma and omega. Both require finite nonnegative K. The returned L is a continuous-time generator, not a sampled operator. For the complex case, omega*dt must be strictly less than pi to recover the intended continuous-time rate using the principal logarithm. Negative-real, zero and unstable map cases are addressed by the separate existing threshold unit tests.

## Independent comparison

The script generates F with `scipy.linalg.expm`, evaluates the package predictions via `threshold_diagnostics`, and compares them to independently evaluated quantities:

- R_A versus the maximum of the logarithmic derivative of A(t) = ||exp(tL)||_F^2. The derivative is computed as 2 times the Frobenius inner product of P and L P, divided by ||P||_F^2, with P=exp(tL). Its sign equals the sign of A'(t). A time grid and bounded local refinement are used; this is a numerical oracle, not a global proof. For real spectra the search extends to 12 slow decay times; for complex spectra one full period of the normalized derivative is sufficient. The threshold itself is a tangent zero, not strict positive growth.
- R_sigma versus the sign of sigma_max(F)-1, computed with a direct singular-value decomposition.

The parameter grid uses gamma1=1 with gamma2 in [1.02, 1.1, 1.5, 2, 4, 10, 100], or gamma=1 with beta in [.05, .2, .5, 1, 2, 5, 10]. For each, K/Kc spans [0, .25, .5, .85, .99, 1, 1.01, 1.2, 2, 4]. Three sampling intervals (.02, .1, .5 divided by the maximum rate/frequency) produce 420 maps from 140 generators. Analytic Kc values select the grid and independently check numerical R_A; neither ground-truth label is obtained from Kc. Signs use a dimensionless tolerance of 1e-8, and the derivative is divided by the fastest rate for classification. Boundary cases are counted separately. The plot is a denser illustrative sweep of two parameter choices, not the entire grid.

## Interpretation

R_A and R_sigma are distinct. The former detects a delayed increasing interval of the isotropic variance-response kernel, not necessarily a peak above its initial value. The latter detects one-step optimal Euclidean growth for a stable map. Neither establishes physical non-normality from an estimated operator. This experiment validates the exact reduced mathematical diagnostics; finite-data recovery, high-dimensional plane extraction and empirical model adequacy require separate experiments.

## Next experiment

See [high-dimensional stochastic recovery](README_recovery.md) for the finite-data benchmark.
