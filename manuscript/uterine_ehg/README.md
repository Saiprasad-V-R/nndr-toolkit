# Uterine electrohysterogram (EHG)

[Manuscript index](../README.md) · [NNDR user documentation](../../docs/quickstart.md)

## Scope

Historical methodology from the original manuscript. Full revised cohort numerical reproduction is not provided here.

## Data and analysis

[Icelandic 16-electrode Electrohysterogram Database](https://physionet.org/content/ehgdb/1.0.0/), WFDB physical signals and contraction annotations. Original representative recording `ice012_p_1of4`, EHG9–EHG12; original cohort 123 recordings/45 subjects. Read calibrated signals with `physical=True`, convert mV to microvolts (do not multiply by the raw MAT conversion factor), filter 0.08–2 Hz, resample to 10 Hz. Original windows 60 s, step 5 s, shrinkage 0.1. Activity reference is the channel-mean Hilbert magnitude. Original high-activity and baseline groups use the upper 20% and lower 50% of that envelope per recording, with recording-level Spearman correlation and Cliff's delta, then subject medians. These activity definitions are not direct mechanical contraction measurements.

For new analyses, specify preprocessing, sampling interval, ridge and admissibility rules. Original cohort counts and settings are historical descriptors, not newly verified revised results.
