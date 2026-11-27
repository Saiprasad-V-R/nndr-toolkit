# Contributing to NNDR

Please keep scientific changes, documentation changes and manuscript reproduction changes distinguishable in commits and pull requests.

## Development setup

```bash
git clone https://github.com/Saiprasad-V-R/nndr-toolkit.git
cd nndr-toolkit
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pip check
python -m pytest -W error::RuntimeWarning
```

The project supports Python 3.9+. The included macOS constraints document one validated environment, not a cross-platform test matrix. MNE is optional and only needed for the EDF reproduction route.

## Project layout

| Directory | Role |
|---|---|
| `src/nndr/` | Scientific implementation: high-level inference, extraction, thresholds, estimation, gains and statistical controls. |
| `tests/` | Numerical and input-validation tests. |
| `docs/` | Conceptual guides, mathematical definitions, API reference and limitations. |
| `examples/` | Self-contained scientific use cases, executable scripts and illustrative reference figures. |
| `manuscript/` | Paper-specific validation scripts, archived results, data and methodology. |

## Scientific and coding conventions

- Use explicit `R_A`, `R_sigma`, `Delta_rate` and `beta` where applicable; do not confuse historical multiplier splitting with rate splitting. See the [mathematical framework](docs/mathematical_framework.md).
- Distinguish discrete maps from continuous generators; document sampling and preprocessing. Reject inadmissible inputs instead of silently changing them.
- Use public NNDR routines in examples, document any additional assumptions and preserve missing/invalid diagnostics. Include tests for new numerical behavior.
- Keep numerical outputs reproducible with explicit settings and seeds. Follow the [figure style](examples/FIGURE_STYLE.md) and keep interpretation outside panels. Do not alter frozen manuscript references to match new estimates.
- Update the [API reference](docs/api_reference.md) and relevant [limitations](docs/limitations.md) for changes in behavior. Use NumPy-style docstrings.

## Reporting issues

Provide a minimal reproducible script, expected and observed behavior, package version or commit, Python and dependency versions, and (if relevant) model or data dimensions, sampling interval and estimator settings. Do not upload confidential measurements.
