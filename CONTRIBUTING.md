# Contributing

Thanks for your interest in improving NNDR.

## Development setup

```bash
git clone https://github.com/Saiprasad-V-R/nndr-toolkit.git
cd nndr-toolkit
pip install -e ".[dev]"
```

## Running the tests

```bash
pytest
```

All tests live in `tests/` and should pass before a pull request is opened. If
you add a feature, add a test that covers it.

## Project layout

```
src/nndr/
  core.py        high-level inference API (analyze, rolling_analyze, ...)
  methods.py     the three plane-extraction methods (M1, M2, M3)
  reduce2d.py    reduced diagnostics (Delta, K, Kc, R)
  ridge.py       operator estimation
  observables.py plane-invariant correlation
  simulate.py    VAR(1) simulation
  generator.py   synthetic operator families (benchmarks)
  benchmark.py   validation harness
  plotting.py    figures from benchmark results
  utils.py       linear-algebra helpers
tests/           pytest suite
examples/        runnable usage scripts
```

## Style

- Keep the core inference path (`core.py`) dependency-light: NumPy only.
- Match the paper's terminology: `R` is a parameter, "pseudo-critical" is an
  adjective for the threshold/regime, methods are M1/M2/M3.
- Add NumPy-style docstrings to public functions.

## Reporting issues

Please include a minimal reproducing example and the versions of `nndr`, NumPy,
and Python you are running.
