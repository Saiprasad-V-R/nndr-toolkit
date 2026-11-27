# Changelog

Notable changes to NNDR. Published snapshots are retained; see the [release history](https://github.com/Saiprasad-V-R/nndr-toolkit/releases).

## Unreleased — scientific usability redesign

- Separate manuscript-specific data, scripts, validation figures and methodology under `manuscript/`.
- Add six independently runnable, illustrated scientific tutorials covering stable amplification, real and complex spectra, network inference, time-varying dynamics, CSV analysis and uncertainty.
- Standardize example figures using a shared publication-style plotting module and export PNG, PDF and SVG.
- Reorganize the user guide, mathematical framework, API navigation and README; correct incompatible math macros.
- Replace two introductory scripts with the expanded CSV and moving-window tutorials; preserve historical versions in Git.
- Update software citation metadata; the next archival DOI remains pending collaborator review.

The scientific API remains 2.0.0. No new empirical results or revised mathematical formulas are introduced by this documentation redesign.

## [2.0.0](https://github.com/Saiprasad-V-R/nndr-toolkit/releases/tag/2.0.0) — 2026-11-26

- Correct real and complex variance-response thresholds and add the distinct discrete one-step ratio `R_sigma`; `R` defaults to `R_A`.
- Add observed adjacent-pair estimation, relative ridge, trace-preserving shrinkage and directed M3 ordering.
- Expose scientific geometry, dynamical gains, model-adequacy diagnostics, phase surrogates, normal references and bootstrap functions.
- Add controlled synthetic threshold and finite-data recovery tests.
- Validate the representative seizure EEG evaluation against archived numerical results, with preserved finite-value masks.
- Add technical documentation and a runnable end-to-end tutorial.
- Verify 134 tests from a clean wheel installation of the preceding scientific implementation on macOS Apple Silicon, Python 3.9.6, before the version-only bump to 2.0.0. Cross-platform CI coverage has not been established.

## [v1.0.0](https://github.com/Saiprasad-V-R/nndr-toolkit/releases/tag/v1.0.0) — 2026-07-16

Original public toolkit snapshot: high-level estimation and rolling analysis, three plane-extraction methods, historical reduced threshold conventions, synthetic benchmark helpers and introductory scripts. See the [migration guide](THRESHOLD_MIGRATION.md) before comparing results with the current API.
