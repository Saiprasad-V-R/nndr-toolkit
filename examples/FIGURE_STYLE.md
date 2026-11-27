# Scientific figure style

All examples use the shared [`_figure_style.py`](_figure_style.py) module. Call `configure_style()` before creating figures and use `format_axis()` for ordinary linear axes. Keep categorical, node-index and logarithmic locators appropriate to the data. Use `panel_label()` with coordinates adjusted to avoid overlaps. Export with `save_figure()` (550-dpi PNG, SVG and PDF).

- Figures show evidence: data, curves, markers, units, panel identifiers, essential reference lines and short legends. Neutral panel titles are useful where the plotted quantity is otherwise ambiguous.
- Captions and the README explain the scientific interpretation, fixed parameters, comparison protocol and limitations. Put exact numerical estimates in tables and machine-readable outputs, not callout boxes or bars labelled with redundant digits.
- Keep shared axes and scales for direct comparisons; preserve the original numerical results and disclose any clipping or normalization. Do not smooth or hide data to improve appearance.
- Use the shared serif/Computer Modern font settings, inward ticks, linewidths and export parameters. Use colors consistently within a comparison; supplement color with markers or line styles when needed. Neutral grey may distinguish normal or reference curves. The supplied palette is available in the style module.
- Inspect the exported figure at its intended display size; retain the numerical data and plotting script.

The tutorial figures are educational examples, distinct from the frozen manuscript figures under `manuscript/`.
