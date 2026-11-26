# Figure 12 — frozen CHB01_03 representative recording

The authoritative source is the supplied CHB01_03.zip (26 November 2026). The three CSV files, plotting scripts, style module, captions and reference PDFs are copied byte-for-byte. Figure reproduction uses these files directly, without refitting or changing the scientific values. Both figures are four panels: mean EEG, RA (main) or Rsigma (supplement), entropy support, discrete multiplier splitting. The display limits intentionally match the supplied script: mean EEG +/-250 uV, RA 0--4, Rsigma 0--9, support 0--0.8, splitting 0--1. Values outside these ranges remain in the CSV, and raw isolated values are not imputed. The original caption discloses clipping; retain it. Thin lines are raw; thick lines are five-endpoint centered medians within valid segments (singleton segments have no median overlay).

## Reproduce the archived figures

From the repository root (with the validated virtual environment active):

```sh
python reproducibility/seizure/representative/scripts/plot_figure12_RA_main.py
python reproducibility/seizure/representative/scripts/plot_figure12_chb01_03_Rsigma_supp.py
```

Outputs are in `representative/figures/`. PDFs are vector plots; typography may differ slightly with platform font availability. Reference originals are in `reference_figures/`. The plots use 201 endpoints -160 to +240 s, with 2 s steps, 40 s windows; the axis extends to +250 s. The EEG waveform also ends at +240 s. No extrapolation.

## Independent EDF reconstruction and parity audit (not yet verified on the original EDF)

```sh
python -W error::RuntimeWarning reproducibility/seizure/representative/evaluate_edf.py --edf ~/Desktop/nndr-seizure-data/chb01_03.edf --out ~/Desktop/nndr-seizure-reference-audit
```

Uses the core16 channel order, first occurrence of normalized channel names, 60 Hz notch Q=30, fifth-order Butterworth 0.5--15 Hz zero-phase bandpass, default SciPy polyphase resampling 256->32 Hz, within-window centering, relative ridge 0.003, trace shrinkage 0.05. These detailed preprocessing choices and the 12-restart/200-iteration M2 with seed 0 come from the earlier Stage-6 40 s sensitivity code. Uses NNDR package fitting and corrected thresholds, historical M2 multistart adapter, and current directed M3. The exact single-event evaluation script that produced the archive was NOT included. M3 ordering, gates and preprocessing may differ from that missing implementation. The output also records entropy support for the raw commutator basis to help diagnose any M3 support discrepancy; it is not selected by fitting to the archive.

Figure-specific entropy support exp(-sum(p log p))/N differs from the package's L1 support. Discrete multiplier splitting differs from Delta_rate; it remains defined for complex pairs with nonzero trace. Neither is silently redefined in the package. The audit retains support/splitting in reduced-unstable windows as the archived CSV does. It compares all eight raw diagnostics and rho_F by exact timestamp, finite masks, absolute error and allclose (rtol=1e-6, atol=1e-7), plus waveform maximum absolute difference (1e-4 uV tolerance). Tolerances are engineering choices, not published inferential thresholds. An audit failure is not a reason to overwrite or tune the reference results. The candidate generates separate figures even on failure; do not confuse them with the archived figures.

The EDF is not bundled; the previous download_data.py obtains it from PhysioNet. The original user-supplied metadata includes local source paths, retained for provenance. The supplied CSVs include derived EEG data; before public upload confirm dataset attribution and redistribution requirements using existing DATA_LICENSE.md. This is a descriptive single seizure, not a detection/prediction validation.
