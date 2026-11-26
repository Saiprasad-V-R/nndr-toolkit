# Quick start

[Home](../README.md) · [Tutorial](tutorial.md) · [API reference](api_reference.md)

## Install

NNDR 2.0.0 is the current published release. To install just the package, with the extras required for the examples, run the following in an activated virtual environment:

```bash
python -m pip install "nndr-toolkit[dev] @ git+https://github.com/Saiprasad-V-R/nndr-toolkit.git@2.0.0"
```

To obtain the repository tutorials and reference data, clone the repository and check out the fixed release tag, as shown below. Omit the detached checkout if you want the current default `main` branch. The [historical v1.0.0 snapshot](https://github.com/Saiprasad-V-R/nndr-toolkit/tree/v1.0.0) uses different conventions.

```bash
git clone https://github.com/Saiprasad-V-R/nndr-toolkit.git
cd nndr-toolkit
git switch --detach 2.0.0
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pip check
```

For Windows activate with `.venv\Scripts\activate`. For the validated macOS Apple Silicon Python 3.9.6 environment, use the included constraints when installing: `python -m pip install -e '.[dev]' -c requirements-validated-macos.txt`. The constraints describe one tested environment, not every platform. For EDF reproduction also install `-r reproducibility/seizure/requirements.txt`. The package import is `nndr`; its distribution name is `nndr-toolkit`.

## Known discrete operator

```python
import numpy as np
import nndr
F = np.array([[0.8, 0.6], [0.0, 0.5]])
res = nndr.analyze_operator(F, method='M3', dt=0.1)
print(res.R_A, res.R_sigma, res.diagnostics['rhoGamma'])
```

For a continuous generator `L`, explicitly sample it first: `F = expm(L * dt)` (import `expm` from `scipy.linalg`). Rates derived from a principal log may alias if sampling is too slow.

## Observations

```python
import numpy as np
import nndr
X = np.load('recording.npy')  # provide your own file, N channels by T samples
X = X - X.mean(axis=1, keepdims=True)  # optional, explicit centering
res = nndr.analyze(X, method='M3', dt=0.01, lam_rel=0.003)
print(res.summary(), res.R_sigma)
```

At least two channels and enough observations for NNDR plane extraction are required; the high-level validator also rejects N > T and T < 3. Do not silently transpose, interpolate or rescale channels. The package fits a homogeneous linear predictor with no intercept. Demeaning can be useful for local fluctuation dynamics, but is not automatic. In a held-out analysis estimate centering on training samples only. When sampling is irregular, establish a justified regular grid or use another appropriate estimator first.

## Rolling windows

```python
records = nndr.rolling_analyze(X, window=400, step=20, method='M3',
                               dt=0.01, lam_rel=0.003)
end = np.array([row['t_end'] for row in records]) * 0.01
ra = np.array([row['R_A'] for row in records])
rs = np.array([row['R_sigma'] for row in records])
```

`window` and `step` are in samples. `t_end` is an exclusive index; the plotted boundary time is `t_end * dt` if the first sample is at zero. `t_center` is an integer index. Rolling windows are not independent observations. Offline zero-phase filtering is not causal. `rolling_analyze` does not center each window; see the tutorial for explicit window-wise centering. Retain missing values and report their counts. Do not use `ok` as a universal adequacy flag.
