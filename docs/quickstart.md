# Quick start: your first NNDR analysis

[Documentation](index.md) · [Understanding NNDR](understanding_nndr.md) · [User guide](user_guide.md)

## 1. Install

Use Python 3.9 or newer. For the fixed published scientific release, with Git installed:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install "nndr-toolkit[dev] @ git+https://github.com/Saiprasad-V-R/nndr-toolkit.git@2.0.0"
python -c "import nndr; print(nndr.__version__)"
```

Expected version: `2.0.0`. On Windows activate using `.venv\Scripts\activate`. The distribution is `nndr-toolkit`; the import is `nndr`. PyPI publication is not required.

To run repository examples, clone the repository and install from the checkout instead:

```bash
git clone https://github.com/Saiprasad-V-R/nndr-toolkit.git
cd nndr-toolkit
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pip check
```

If already inside a checkout, do not clone it again or switch branches. The current checkout contains the reorganized manuscript resources; the fixed tag `2.0.0` retains its earlier layout, so follow the documentation inside that tag for exact snapshot reproduction. For the validated Apple Silicon macOS/Python 3.9.6 environment, use `python -m pip install -e '.[dev]' -c requirements-validated-macos.txt`. This is one tested environment, not a universal dependency lock. MNE is needed for the empirical EEG evaluation in the [reproduction repository](https://github.com/Saiprasad-V-R/nndr-reproducibility).

## 2. Understand one small system

```python
import numpy as np
import nndr

F = np.array([[0.8, 0.6], [0.0, 0.5]])
r = nndr.analyze_operator(F, method='M3', dt=1.0)
print(r.diagnostics['rhoGamma'])
print(r.R_A, r.R_sigma)
```

Expected values, approximately: spectral radius 0.8, `R_A = 0.812`, `R_sigma = 1.155`. The map is stable but allows one-step norm growth. The two ratios answer distinct questions; see the [conceptual guide](understanding_nndr.md). A continuous generator must be sampled first with `scipy.linalg.expm(L * dt)`.

## 3. Analyze a multivariate record

```python
L = nndr.complex_generator(1.0, 2.0, 1.5)
F, Q_true = nndr.embedded_map(L, dimension=12, dt=0.1, seed=1)
X = nndr.simulate_trajectory(F, samples=5000, seed=2)
r = nndr.analyze(X, method='M3', dt=0.1,
                 lam_rel=0.003, shrinkage=0.04)
print(r.summary())
print(r.R_A, r.R_sigma, r.support)
```

Replace `X` with your own finite real array of shape `(channels, samples)`, at least two channels, at least three samples and no more channels than samples. Use the actual sampling interval and justified preprocessing. Centering is not automatic: the fitted predictor has no intercept. The example parameters are illustrative, not universal. To compare methods, fit once and use `analyze_operator` for each method, as shown in the [user guide](user_guide.md).

## 4. Track changes in time

```python
rows = nndr.rolling_analyze(X, window=1000, step=250,
                            method='M3', dt=0.1, lam_rel=0.003,
                            shrinkage=0.04)
end = np.array([row['t_end'] for row in rows]) * 0.1
ra = np.array([row['R_A'] for row in rows])
rs = np.array([row['R_sigma'] for row in rows])
```

Window and step are in samples. `t_end` is an exclusive index, and the time is the trailing boundary if sampling begins at zero. The routine does not center each window. Overlapping windows are dependent; retain missing values and do not treat `ok` as an adequacy test. The [tutorial](tutorial.md) shows how to center and fit windows explicitly.

## 5. Continue with a complete workflow

Run the existing synthetic tutorial from the repository root:

```bash
python examples/tutorial_nndr.py --out results/tutorial
```

This example needs no external data. Read the [user guide](user_guide.md) for your use case and [validation and uncertainty](validation_and_uncertainty.md) before interpreting results from experimental records.
