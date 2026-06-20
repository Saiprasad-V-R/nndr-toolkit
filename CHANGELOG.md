# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and this project adheres to
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Input-validation guards (`nndr.validation`) with clear, actionable error
  messages for common mistakes: 1-D input, transposed data, NaN/Inf values,
  windows longer than the series, and non-square operators. The guards fire
  through the public API (`analyze`, `rolling_analyze`, `fit_operator`).
- Continuous integration via GitHub Actions: the test suite runs on Python
  3.9–3.12 and a wheel is built on every push and pull request.

## [0.1.0] — 2026

Initial public release.

### Added
- High-level, inference-first API in `nndr.core`:
  - `analyze` — estimate the local operator from a time series and compute the
    reduced diagnostics (`R`, `Delta`, `K`, support, response plane).
  - `rolling_analyze` — the same in a sliding window, for non-stationary data.
  - `analyze_operator` — diagnostics from a known operator (no estimation).
  - `fit_operator` — ridge-fit the one-step operator, with optional
    trace-preserving shrinkage.
  - `NNDRResult` dataclass with a `summary()` helper.
- Three plane-extraction methods in `nndr.methods`:
  - M1 — eigenbasis-SVD baseline extraction.
  - M2 — optimization-based directed-coupling extraction (recommended).
  - M3 — commutator-based plane extraction (independent cross-check).
- Reduced diagnostics in `nndr.reduce2d`, including the closed-form threshold
  `Kc(Delta)`.
- Validation harness (`nndr.benchmark`) and plotting helpers (`nndr.plotting`),
  available via the optional `benchmark` extra.
- Test suite (16 tests) covering diagnostics, methods, and the high-level API.
- Three runnable examples under `examples/`.
- Packaging via `pyproject.toml` (hatchling), MIT license, and `CITATION.cff`.
