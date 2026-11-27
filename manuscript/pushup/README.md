# Push-up inertial recording

Original wearable inertial recording used for Figure 16 of the paper.

**Sensor.** Apple Watch, Core Motion **user acceleration** (gravity already
removed by Core Motion), reported in units of *g* (≈ 9.8 m s⁻²).

**Protocol.** Rhythmic push-ups performed on two small unstable, deformable
elastic fitballs (one under each closed fist, placed parallel to the body),
following the Logic Workout protocol (https://logicworkoutapp.com), which
exploits controlled instability and the "reactive falling effect."

## File

`raw_samples_Pushup_duo_27Dec2025..csv`

CSV with a header row and the following columns:

| column      | type    | units | description                          |
|-------------|---------|-------|--------------------------------------|
| `timestamp` | float/ISO | s   | sample time; used to infer the sampling interval |
| `accX`      | float   | g     | Core Motion user acceleration, x-axis |
| `accY`      | float   | g     | Core Motion user acceleration, y-axis |
| `accZ`      | float   | g     | Core Motion user acceleration, z-axis |

The analysis subtracts the residual per-channel mean before fitting; no other
calibration is applied.

## How it is used

This recording is the input for the push-up analysis (Figure 16 of the paper).
See [the consolidated reproducibility guide](../README.md) for the revised methodology and scope. The analysis code is available from the authors on request.

## License / use

This recording is released with the toolkit for reproducibility of Figure 16.
Please cite the accompanying paper if you use it.

## Revised manuscript methodology (not independently rerun)


The included [Apple Watch data](raw_samples_Pushup_duo_27Dec2025..csv) (`raw_samples_Pushup_duo_27Dec2025..csv`) contain timestamp and `accX`, `accY`, `accZ` user acceleration in g, with gravity already removed. The final revised push-up workflow is described here using the authors' `README_pushup.md` and `fig15_pushup_revised_final.py`; the standalone script is not distributed in this selected reproduction folder.

- Primary signal: demeaned raw acceleration, without per-axis standardization or forced stability rescaling. Classify adjacent pairs by the activity at their predictor time, using a causal trailing 0.5 s mean of acceleration magnitude and the upper 15% as high activity.
- Fit separate low/high operators with relative ridge 0.003, shrinkage 0.04, and M2/M3. Report full and reduced spectral radii, singular values and both corrected ratios.
- Offline sensitivity: phase-adaptive three-harmonic removal of voluntary movement, residual bands 1–15 Hz and 1–24 Hz.
- Class-conditioned uncertainty: 2000 circular moving-block bootstrap replicates with 5 s blocks, preserving original pairs and labels without constructing cross-block transitions. Stationary VAR residual-vector and common-phase Fourier nulls; circular-shift timing is secondary.
- Separate registered moving-window test: raw demeaned acceleration, M3, 1 s window, 0.1 s step; measure the fraction satisfying `rho(Gamma) < 1 < sigma_max(Gamma)`. Compare with 500 time-varying spectrum- and total-variance-matched normal null realizations and 2000 circular 5 s bootstrap replicates. Use a one-sided Monte Carlo rank p-value.

Reference data checks from the final push-up README: 2252 samples, approximately 50.0854 Hz, 1913 low and 338 high pairs. This is a single-participant illustration. The push-up script and complete revised outputs were not independently rerun as part of this repository update; full reproduction is available only with the authors' final analysis source. Do not confuse the class-conditioned ratios with the separate moving-window statistic.
