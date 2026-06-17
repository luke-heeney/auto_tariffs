# Cost-Side Robustness Execution Sequence

This document fixes ownership for the exchange-rate identification robustness workstream and names the exact artifacts each step should produce.

## Scope

The workstream adds the following deliverables on top of the existing cost-side pipeline:

1. placebo regressions for exchange-rate identification,
2. lag-current-future timing regressions,
3. canonical-sample pass-through diagnostics,
4. leave-one-country-out checks,
5. alternative exposure definitions,
6. price/markup mechanism decompositions,
7. vehicle-type heterogeneity diagnostics, and
8. reduced-form figures for the cost channel.

The canonical coauthor-facing artifact is now `cost_side/docs/full_robustness_report.pdf`, built by `bash cost_side/build_robustness_report.sh`.

The existing paper-facing baseline table remains owned by `cost_side/cost_reg.R`. The counterfactual-facing elasticity-interaction coefficients remain owned by `cost_side/cost_reg_elas.R`. The new workstream is additive and writes separate outputs rather than silently replacing current baseline artifacts.

## Execution Order

### Canonical report build

Run:

1. `bash cost_side/build_robustness_report.sh`

This wrapper executes the full robustness pipeline, regenerates the memo tables and figures, and compiles `cost_side/docs/full_robustness_report.pdf`.

### Underlying script order

1. Run `python cost_side/build_cost_side_panel.py`
   This is the only upstream data-construction step for the new workstream.

2. Run `Rscript cost_side/cost_reg_placebo.R`
   This produces the foreign-assembled, future-exchange-rate, and conditional current-plus-future placebo regressions.

3. Run `Rscript cost_side/cost_reg_timing_triplet.R`
   This compares the current-plus-future timing diagnostic to lag-current-future timing specifications.

4. Run `Rscript cost_side/cost_reg_robustness.R`
   This produces canonical-sample robustness tables, including the exchange-rate main-effect and elasticity-interaction diagnostics.

5. Run `Rscript cost_side/cost_reg_leave_one_country_out.R`
   This produces the country-omission robustness summaries and coefficient plot.

6. Run `Rscript cost_side/plot_cost_reduced_form.R`
   This produces the reduced-form visualization of cost changes against exchange-rate shocks for high- and low-exposure domestic vehicles.

7. Run `Rscript cost_side/cost_reg_price_markup_decomp.R`
   This produces the recovered-cost / observed-price / recovered-markup decomposition tables for the baseline and forward-RER placebo designs.

8. Run `Rscript cost_side/cost_reg_alt_exposure.R`
   This produces the alternative-exposure first-difference robustness table.

9. Run `Rscript cost_side/cost_reg_vehicle_type.R`
   This produces vehicle-type-specific pass-through diagnostics.

10. Run `Rscript cost_side/plot_high_exposure_series.R`
   This produces the high-exposure year-series and first-difference decomposition figures.

11. Run `python cost_side/render_full_robustness_report_assets.py`
   This produces the report-specific summary tables.

12. Run `python cost_side/render_cost_side_robustness_note_assets.py`
    This produces the concise review-note tables.

13. Compile `cost_side/docs/full_robustness_report.tex` and `cost_side/docs/cost_side_robustness_note.tex` with `xelatex` twice.
    This produces `cost_side/docs/full_robustness_report.pdf` and `cost_side/docs/cost_side_robustness_note.pdf`.

## File Ownership

### `cost_side/build_cost_side_panel.py`

Owns:

- parsing the raw manufacturer sourcing files,
- attaching plant-country information,
- collapsing duplicate product rows,
- merging costs, characteristics, and exchange rates,
- tagging sample-construction flags, and
- exporting reproducible panel variants for downstream scripts.

Outputs:

- `cost_side/cost_side_panel_all.csv`
- `cost_side/cost_side_panel.csv`
- `cost_side/cost_side_panel_foreign.csv`
- `cost_side/outputs/panel_build_sample_counts.csv`
- `cost_side/outputs/panel_build_diagnostics.csv`
- `cost_side/outputs/source_country_switchers.csv`
- `cost_side/outputs/primary_source_country_conflicts.csv`
- `cost_side/outputs/panel_build_summary.md`

### `cost_side/robustness_helpers.R`

Owns:

- shared panel-loading logic,
- shared regression-frame construction,
- shared first-difference construction, and
- small helpers for coefficient extraction and sample summaries.

Outputs:

- none directly; imported by downstream `R` scripts.

### `cost_side/cost_reg_placebo.R`

Owns:

- domestic baseline placebo comparison sample,
- foreign-assembled placebo regressions,
- standalone future-exchange-rate placebo regressions, and
- conditional current-plus-future placebo regressions that control for `rho_{t-1} x log(RER_t)` while testing `rho_{t-1} x log(RER_{t+1})`.

Decision details:

- The levels future-RER placebo must merge `RER_{t+1}` directly from the current row's `pcOth1_code1` and `year + 1`.
- The levels placebo must not require a next observed make-model row.
- The first-difference future-RER placebo must construct `\Delta \log(RER_{t+1})` directly as `\log(RER_{t+1}) - \log(RER_t)` using the current row's source country. It must not require a next observed make-model row.
- The current-plus-future regressions are the main timing diagnostic because they ask whether `rho_{t-1} x log(RER_{t+1})` predicts current costs after controlling for the contemporaneous `rho_{t-1} x log(RER_t)` term.
- If the exchange-rate file is missing the needed `t+1` country-year values, populate them first with `python cost_side/backfill_exchange_rates_2025.py`. This uses cached IMF values from `cost_side/outputs/exchange_rate_2025_backfill_values.csv` by default; use `--refresh` only to query IMF again.

Outputs:

- `cost_side/outputs/cost_reg_placebo_levels_table.tex`
- `cost_side/outputs/cost_reg_placebo_fd_table.tex`
- `cost_side/outputs/cost_reg_placebo_coefficients.csv`
- `cost_side/outputs/cost_reg_placebo_sample_counts.csv`
- `cost_side/outputs/cost_reg_placebo_notes.md`

### `cost_side/cost_reg_timing_triplet.R`

Owns:

- lag-current-future timing regressions in levels and first differences,
- comparison with the current-plus-future timing diagnostic on the same matched sample, and
- timing-triplet coefficient and sample-count summaries.

Outputs:

- `cost_side/outputs/cost_reg_timing_triplet_levels_table.tex`
- `cost_side/outputs/cost_reg_timing_triplet_fd_table.tex`
- `cost_side/outputs/cost_reg_timing_triplet_coefficients.csv`
- `cost_side/outputs/cost_reg_timing_triplet_sample_counts.csv`
- `cost_side/outputs/cost_reg_timing_triplet_notes.md`

### `cost_side/cost_reg_robustness.R`

Owns:

- canonical baseline pass-through regressions,
- a specification with an unrestricted exchange-rate main effect alongside the exposure interaction, and
- canonical elasticity-interaction diagnostics.

Outputs:

- `cost_side/outputs/cost_reg_robustness_canonical_table.tex`
- `cost_side/outputs/cost_reg_robustness_rer_main_table.tex`
- `cost_side/outputs/cost_reg_robustness_elas_table.tex`
- `cost_side/outputs/cost_reg_robustness_coefficients.csv`
- `cost_side/outputs/cost_reg_robustness_rer_main_coefficients.csv`
- `cost_side/outputs/cost_reg_robustness_sample_counts.csv`
- `cost_side/outputs/cost_reg_robustness_notes.md`

### `cost_side/cost_reg_leave_one_country_out.R`

Owns:

- leave-one-country-out versions of the canonical domestic regression sample,
- the summary CSV for omitted-country checks, and
- the coefficient comparison plot.

Outputs:

- `cost_side/outputs/leave_one_country_out_coefficients.csv`
- `cost_side/outputs/leave_one_country_out_sample_counts.csv`
- `cost_side/outputs/leave_one_country_out_eta.png`
- `cost_side/outputs/leave_one_country_out_notes.md`

### `cost_side/plot_cost_reduced_form.R`

Owns:

- the reduced-form high-versus-low exposure plot built from the canonical domestic sample,
- the residualized and binned plotting data, and
- the short note describing the plot construction.

Outputs:

- `cost_side/outputs/cost_reduced_form_binned.csv`
- `cost_side/outputs/cost_reduced_form_fx.png`
- `cost_side/outputs/cost_reduced_form_notes.md`

### `cost_side/cost_reg_price_markup_decomp.R`

Owns:

- the baseline decomposition of the exchange-rate exposure term across recovered marginal costs, observed prices, and recovered markups, and
- the same decomposition for the forward-RER placebo.

Outputs:

- `cost_side/outputs/cost_reg_price_markup_decomp_table.tex`
- `cost_side/outputs/cost_reg_price_markup_forward_placebo_table.tex`
- `cost_side/outputs/cost_reg_price_markup_decomp_coefficients.csv`
- `cost_side/outputs/cost_reg_price_markup_decomp_notes.md`

### `cost_side/cost_reg_alt_exposure.R`

Owns:

- first-difference robustness checks using alternative exposure definitions.

Outputs:

- `cost_side/outputs/cost_reg_alt_exposure_table.tex`
- `cost_side/outputs/cost_reg_alt_exposure_coefficients.csv`
- `cost_side/outputs/cost_reg_alt_exposure_notes.md`

### `cost_side/cost_reg_vehicle_type.R`

Owns:

- vehicle-type-specific pass-through diagnostics for cars, trucks, SUVs, and vans.

Outputs:

- `cost_side/outputs/cost_reg_vehicle_type_table.tex`
- `cost_side/outputs/cost_reg_vehicle_type_sample_counts.csv`
- `cost_side/outputs/cost_reg_vehicle_type_coefficients.csv`

### `cost_side/plot_high_exposure_series.R`

Owns:

- year-series visuals for recovered costs, observed prices, and exchange rates on the high-exposure sample, and
- residualized first-difference visuals for recovered costs, observed prices, and recovered markups on the same high-exposure sample.

Outputs:

- `cost_side/outputs/high_exposure_series_values.csv`
- `cost_side/outputs/high_exposure_series_mc_price_rer.png`
- `cost_side/outputs/high_exposure_fd_binned.csv`
- `cost_side/outputs/high_exposure_fd_decomp.png`
- `cost_side/outputs/high_exposure_plot_notes.md`

### `cost_side/render_full_robustness_report_assets.py`

Owns:

- the report-level summary identification table, and
- the report-level leave-one-country-out summary table.

Outputs:

- `cost_side/outputs/full_robustness_main_table.tex`
- `cost_side/outputs/leave_one_country_out_report_table.tex`

### `cost_side/render_cost_side_robustness_note_assets.py`

Owns:

- compact review-note tables for sample construction, robustness registry, timing checks, exposure construction, alternative exposure definitions, vehicle-type diagnostics, and price/markup decomposition.

Outputs:

- `cost_side/outputs/cost_side_note_sample_table.tex`
- `cost_side/outputs/cost_side_note_registry_table.tex`
- `cost_side/outputs/cost_side_note_timing_table.tex`
- `cost_side/outputs/cost_side_note_exposure_table.tex`
- `cost_side/outputs/cost_side_note_alt_exposure_table.tex`
- `cost_side/outputs/cost_side_note_vehicle_type_table.tex`
- `cost_side/outputs/cost_side_note_decomp_table.tex`

## Decision Rules

- The domestic baseline for placebo, reduced-form work, and cost-side robustness is the canonical domestic panel written to `cost_side/cost_side_panel.csv`.
- The foreign placebo should use the same variable construction as the domestic baseline, but on the foreign-assembled source-stable sample.
- The levels future-RER placebo should use direct country-year exchange-rate availability. Missing future exchange rates should reflect missing `t+1` country-year data, not missing next-row observations within make-model.
- Duplicate domestic product-year rows may be collapsed only when their directly observed raw primary source country agrees. Conflicting raw primary source-country rows are excluded from the canonical domestic panel and recorded in `cost_side/outputs/primary_source_country_conflicts.csv`.
- The workstream should not overwrite the current baseline coefficient file `cost_side/outputs/cost_reg_elas_primary_spec_coeffs.csv`.
- Any change to a sample definition must be documented in `cost_side/docs/panel_build_decisions.md` before it is treated as a paper-facing robustness result.
- The full robustness report must exclude source-country-FE variants and lagged-RER timing-triplet variants.
