# NNDR — Non-Normal Directional Response inference
**V. R. Saiprasad, V. Troude, D. Sornette**

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)



A Python toolkit for inferring **non-normal amplification geometry** from multivariate time series.

Stable high-dimensional systems can still produce large finite-time responses when their local dynamics are *non-normal*: eigenvalues govern long-term decay, but non-orthogonal eigenvectors can route perturbations into directions that transiently amplify. NNDR estimates a window-local linear operator from data, extracts the two-dimensional plane that organizes this response, and reports a small set of interpretable diagnostics.

The central diagnostic is the **normalized reduced non-normality ratio**

```
R = K / Kc(Δ)
```

where `Δ` is the reduced eigenvalue splitting, `K` measures reduced eigenvector non-orthogonality, and `Kc(Δ)` is the two-dimensional reduced threshold. `R < 1` means the inferred geometry lies below the reduced transient-amplification threshold; `R > 1` means it lies above it.

---

## Installation



```bash
git clone https://github.com/Saiprasad-V-R/nndr-toolkit.git
cd nndr-toolkit
pip install -e .
```


The import name is `nndr`:

```python
import nndr
```

---

## Quick start

Apply NNDR to your own recording. Your data should be a 2-D array of shape `(N_channels, T_samples)` — one row per channel/sensor, one column per time sample.

```python
import numpy as np
from nndr import analyze

X = np.load("my_recording.npy")     # shape (N_channels, T_samples)

result = analyze(X, method="M2")
print(result.summary())
# [M2] R=1.84 (above threshold), Delta=0.31, K=3.12, support=0.74

print(result.R)        # normalized reduced non-normality ratio
print(result.Delta)    # reduced eigenvalue splitting
print(result.K)        # reduced non-normality index
print(result.Q)        # the response plane [r_hat, n_hat], shape (N, 2)
```

`analyze` does three things under the hood: it fits a one-step linear operator to your data by ridge regression, extracts the response plane, and computes the reduced diagnostics. The fitted operator is available as `result.Fhat`.

### Time-varying data (rolling window)

For recordings whose local dynamics drift, slide a window across the series and get one result per window:

```python
from nndr import rolling_analyze

records = rolling_analyze(X, window=400, step=20, method="M2")

import numpy as np
t = [r["t_center"] for r in records]
R = [r["R"] for r in records]
# plot R against t to see how the reduced geometry evolves
```

Each record carries `t_start`, `t_end`, `t_center`, and the diagnostics `R`, `Delta`, `K`, `support`, plus an `ok` flag that is `False` for degenerate windows you may want to drop.

---

## The three extraction methods

The reduced diagnostics are computed from a 2-D response plane `Q = [r_hat, n_hat]`. Three methods extract that plane; they differ only in *how* the plane is found, after which all diagnostics are identical.

| Key | Name | Notes |
|-----|------|-------|
| `M1` | Eigenbasis-SVD baseline extraction | Transparent reference; fragile when fitted eigenvectors are unstable. |
| `M2` | Optimization-based directed-coupling extraction | **Recommended.** Avoids the eigendecomposition; returns an ordered input/response pair. |
| `M3` | Commutator-based plane extraction | Independent check on M2; uses the symmetric commutator `A Aᵀ − Aᵀ A`. |

In practice, use **M2** as the primary estimator and **M3** as an independent cross-check — agreement between them is evidence that the inferred plane is a property of the dynamics rather than an artifact of one extraction rule.

```python
res2 = analyze(X, method="M2")
res3 = analyze(X, method="M3")
# compare res2.R and res3.R; compare the planes via principal angle
from nndr.utils import principal_angle_deg
print(principal_angle_deg(res2.Q, res3.Q), "degrees")
```

---

## What the diagnostics mean

- **`Delta` (Δ) — reduced eigenvalue splitting.** `|(λ₁ − λ₂)/(λ₁ + λ₂)|` of the reduced 2×2 operator. Near 0 means a nearly degenerate reduced spectrum; near 1 means strongly separated eigenvalues.
- **`K` — reduced non-normality index.** Zero for an orthogonal reduced eigenbasis; grows as the two reduced eigenvectors become more parallel.
- **`Kc(Δ)` — reduced threshold.** The amount of non-orthogonality needed, given `Δ`, to reach the transient-amplification threshold. Derived in Appendix A and Appendix B of the paper.
- **`R = K / Kc(Δ)` — normalized ratio.** A dimensionless control parameter. `R = 1` is the threshold; `R > 1` is the amplification-prone regime. **`R` is a parameter, not a phenomenon** — it controls whether the reduced dynamics can transiently amplify.
- **`support` — reaction-direction support.** `‖r̂‖₁ / (√N ‖r̂‖₂)`, in `(0, 1]`. Near 1 means the reaction direction is spread across many channels; small values mean it is concentrated on a few.

---

## Reproducing the paper's synthetic benchmarks

The validation harness builds known operators, estimates them from finite data, and measures how well each method recovers the plane and `R`. It needs the `benchmark` extra.

```python
import numpy as np
from nndr.benchmark import BenchConfig, run_benchmark

df = run_benchmark(
    N_list=[100],
    M_list=[25],
    T_list=[200],
    kappas=list(np.logspace(0, 2.5, 30)),
    bc=BenchConfig(),
    out_csv="benchmark_results.csv",
)
```

The result is a tidy `pandas` DataFrame (also written to CSV) with one row per (setting, method). Helpers in `nndr.plotting` turn it into figures:

```python
from nndr.plotting import add_R_error_columns, plot_median_vs_x

df = add_R_error_columns(df)
plot_median_vs_x(df, xcol="kappa", ycol="angle_deg",
                 title="Plane error vs anisotropy", ylabel="angle (deg)",
                 outpath="figures/plane_error.png")
```

See [`examples/`](examples/) for runnable scripts.

---

## Reproducing the paper's empirical figures

The four empirical applications (uterine EHG, seizure EEG, freezing of gait, and
the push-up recording) each use a dataset-specific workflow. Data sources,
selected recordings, channels, preprocessing, window parameters, and the
figure-generation scripts are documented in
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md). The per-figure scripts live in
[`reproducibility/`](reproducibility/); the original push-up recording is under
[`data/pushup/`](data/pushup/).

---

## API overview

```python
from nndr import (
    analyze,            # estimate operator + diagnostics from a time series
    rolling_analyze,    # the same, in a sliding window
    analyze_operator,   # diagnostics from a known operator (no estimation)
    fit_operator,       # ridge-fit the one-step operator only
    NNDRResult,         # the result dataclass
)
```

Lower-level building blocks live in `nndr.methods`, `nndr.reduce2d`, `nndr.ridge`, and `nndr.observables`.

---

## Citation

If you use this toolkit, please cite the accompanying paper:

```bibtex
@article{saiprasad2026nndr,
  title         = {Inferring Non-Normal Amplification Geometry from Multivariate Time Series},
  author        = {Saiprasad, V. R. and Troude, V. and Sornette, D.},
  year          = {2026},
  note          = {Preprint. V. R. Saiprasad and V. Troude contributed equally.},
  eprint        = {arXiv:XXXX.XXXXX},
  archivePrefix = {arXiv},
  primaryClass  = {nlin.CD}
}
```



---

## License

MIT — see [LICENSE](LICENSE).
