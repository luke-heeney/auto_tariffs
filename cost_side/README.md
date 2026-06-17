# cost_side

This directory contains the cost-side regressions used to relate exchange rates, imported-parts exposure, and vehicle costs.

## What is here

- `cost_reg.R` — baseline cost regression script.
- `cost_reg_elas.R` — elasticity-interaction regression script. This is the main script if you want the pass-through coefficients used by `post_est/`.
- `cost_reg_vehicle_type.R` — vehicle-type splits.
- `build_cost_side_panel.py` — scripted panel builder that replaces the notebook-only data-construction step for the robustness workflow.
- `robustness_helpers.R` — shared regression-sample helpers for the robustness scripts.
- `backfill_exchange_rates_2025.py` — reproducible IMF-based backfill for the `2025` column in `processed_data/exchange_rates/exchange_rates.csv`, plus a markdown summary of the populated countries. The default uses cached IMF values in `cost_side/outputs/exchange_rate_2025_backfill_values.csv`; pass `--refresh` only when intentionally refreshing from IMF.
- `cost_reg_placebo.R` — foreign-assembled, future-exchange-rate, and conditional current-plus-future placebo regressions. Both the levels and first-difference future-RER placebos use direct country-year `t+1` exchange-rate terms rather than next observed make-model rows.
- `cost_reg_timing_triplet.R` — lag-current-future timing regressions that check whether the future-RER placebo is proxying for exchange-rate persistence.
- `cost_reg_robustness.R` — canonical-sample pass-through diagnostics, plus a specification that adds a plain exchange-rate main effect alongside the exposure interaction.
- `cost_reg_leave_one_country_out.R` — leave-one-country-out checks for the domestic baseline sample.
- `plot_cost_reduced_form.R` — reduced-form figure for cost changes against exchange-rate shocks.
- `cost_reg_alt_exposure.R` — first-difference checks using alternative imported-parts exposure definitions.
- `cost_reg_price_markup_decomp.R` — recovered-cost, observed-price, and recovered-markup decomposition checks.
- `plot_implied_pass_through.R` — plots implied pass-through against elasticity.
- `run_cost_side.ipynb` — notebook version of the cost-side workflow.
- `cost_side_panel.csv` — slim canonical domestic regression panel.
- `cost_side_panel_all.csv`, `cost_side_panel_foreign.csv` — support panels for diagnostics, descriptives, and placebo checks.
- `docs/` — markdown documentation for sample definitions and the robustness execution sequence.
- `outputs/` — generated tables, figures, and CSVs.

## Recommended workflow

Run the full cost-side downstream workflow from the repository root:

```bash
bash cost_side/make.sh
```

This wrapper builds the cost-side panel, regenerates the primary paper-facing cost regression, regenerates the elasticity-interaction regression outputs, and builds the robustness reports.

The elasticity-interaction regression itself:

1. loads the cost-side panel,
2. merges in product-year elasticities from `post_est/data/derived/product_year_elasticities.csv`,
3. estimates levels and first-difference specifications for the domestic-assembly sample,
4. writes LaTeX tables and diagnostics into `cost_side/outputs/`, and
5. exports the primary-spec coefficients to `cost_side/outputs/cost_reg_elas_primary_spec_coeffs.csv`.

## Robustness workflow

From the repository root:

```bash
python cost_side/build_cost_side_panel.py
python cost_side/backfill_exchange_rates_2025.py
Rscript cost_side/cost_reg_placebo.R
Rscript cost_side/cost_reg_timing_triplet.R
Rscript cost_side/cost_reg_robustness.R
Rscript cost_side/cost_reg_leave_one_country_out.R
Rscript cost_side/plot_cost_reduced_form.R
Rscript cost_side/cost_reg_price_markup_decomp.R
Rscript cost_side/cost_reg_alt_exposure.R
Rscript cost_side/cost_reg_vehicle_type.R
Rscript cost_side/plot_high_exposure_series.R
```

The execution order and sample-definition rules are documented in:

- `cost_side/docs/robustness_execution_sequence.md`
- `cost_side/docs/panel_build_decisions.md`
- `cost_side/docs/rer_literature_benchmark.md`
- `cost_side/docs/full_robustness_report_notes.md`
- `cost_side/outputs/exchange_rate_2025_backfill_summary.md`

To rebuild the full coauthor-facing robustness memo:

```bash
bash cost_side/build_robustness_report.sh
```

This produces `cost_side/docs/cost_side_robustness_note.pdf` as the polished coauthor-facing note and `cost_side/docs/full_robustness_report.pdf` as the longer supporting report.

The robustness report uses cached 2025 IMF exchange-rate backfill values by
default, so the replication workflow does not depend on live network access.
To refresh those values deliberately, run:

```bash
python cost_side/backfill_exchange_rates_2025.py --refresh
```

## Why the coefficient CSV matters

`post_est/results_config*.json` can point to `cost_side/outputs/cost_reg_elas_primary_spec_coeffs.csv` when the counterfactuals use `parts_cost_adjustment.mode = "elasticity_interaction"`.

That file gives the intercept and elasticity-interaction slope used to convert a product's own-price elasticity into a parts-cost pass-through rate during the tariff simulations.

## Expected outputs

The most useful generated files are:

- `outputs/cost_reg_elas_merge_diagnostics.csv`
- `outputs/cost_reg_elas_levels_table.tex`
- `outputs/cost_reg_elas_fd_table.tex`
- `outputs/cost_reg_elas_primary_spec_coeffs.csv`
- `outputs/panel_build_sample_counts.csv`
- `outputs/cost_reg_placebo_coefficients.csv`
- `outputs/cost_reg_timing_triplet_coefficients.csv`
- `outputs/cost_reg_robustness_coefficients.csv`
- `outputs/cost_reg_robustness_rer_main_table.tex`
- `outputs/cost_reg_robustness_rer_main_coefficients.csv`
- `outputs/cost_reg_price_markup_decomp_table.tex`
- `outputs/cost_reg_alt_exposure_table.tex`
- `outputs/cost_reg_vehicle_type_table.tex`
- `outputs/cost_reg_vehicle_type_coefficients.csv`
- `outputs/full_robustness_main_table.tex`
- `outputs/cost_side_note_sample_table.tex`
- `outputs/cost_side_note_registry_table.tex`
- `outputs/cost_side_note_timing_table.tex`
- `outputs/cost_side_note_exposure_table.tex`
- `outputs/cost_side_note_alt_exposure_table.tex`
- `outputs/cost_side_note_vehicle_type_table.tex`
- `outputs/cost_side_note_decomp_table.tex`
- `outputs/high_exposure_series_mc_price_rer.png`
- `outputs/high_exposure_fd_decomp.png`
- `outputs/leave_one_country_out_coefficients.csv`
- `outputs/cost_reduced_form_fx.png`

Treat files under `outputs/` as generated artifacts rather than hand-edited source files.
