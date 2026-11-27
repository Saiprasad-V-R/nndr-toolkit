# Scientific examples

[Repository home](../README.md) · [Documentation](../docs/index.md) · [Manuscript reproducibility](../manuscript/README.md)

These examples teach NNDR through scientific questions. They are independent of the paper-specific data and can be run offline after installing the package and its `dev` extras.

| Example | Scientific question |
|---|---|
| [01 — Why non-normality matters](01_why_non_normality_matters/README.md) | Why can identical stable eigenvalues produce different finite-time responses? |
| [02 — Real and complex spectra](02_real_and_complex_spectra/README.md) | How do spectral family and sampling interval change amplification? |
| [03 — Infer hidden network geometry](03_inference_from_network_data/README.md) | Can node observations recover a planted response plane and its directions? |
| [04 — Time-varying dynamics](04_time_varying_dynamics/README.md) | How does window length affect tracking of changing amplification? |
| [05 — Analyze your own data](05_analyze_your_own_data/README.md) | How can I analyze a multichannel CSV without knowing the governing equations? |
| [06 — Validation and uncertainty](06_validation_and_uncertainty/README.md) | How can I assess whether a fitted amplification estimate is reliable? |

[![Identical stable spectra, different transient responses](01_why_non_normality_matters/figures/comparison.png)](01_why_non_normality_matters/README.md)

[![Real and complex spectral families](02_real_and_complex_spectra/figures/spectral_comparison.png)](02_real_and_complex_spectra/README.md)

[![Infer hidden network amplification geometry](03_inference_from_network_data/figures/network_inference.png)](03_inference_from_network_data/README.md)

[![Track time-varying amplification](04_time_varying_dynamics/figures/time_varying_dynamics.png)](04_time_varying_dynamics/README.md)

[![Analyze a multichannel recording](05_analyze_your_own_data/figures/recording_analysis.png)](05_analyze_your_own_data/README.md)

[![Model validation and uncertainty](06_validation_and_uncertainty/figures/validation_and_uncertainty.png)](06_validation_and_uncertainty/README.md)

For the existing end-to-end tutorial, see [the tutorial guide](../docs/tutorial.md) and [its runnable script](tutorial_nndr.py). Manuscript-specific validation and figures are maintained separately in [manuscript/](../manuscript/README.md).
