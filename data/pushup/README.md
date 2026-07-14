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
See [`../../REPRODUCIBILITY.md`](../../REPRODUCIBILITY.md) (Section 4) for the full
preprocessing, per-class fitting, cycle-removal robustness check, and timing
test. The analysis code is available from the authors on request.

## License / use

This recording is released with the toolkit for reproducibility of Figure 16.
Please cite the accompanying paper if you use it.
