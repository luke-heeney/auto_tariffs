# Full Robustness Report Notes

This file documents the intended scope of `cost_side/docs/full_robustness_report.tex`.

## Included

- baseline levels and first-difference cost regressions
- forward-RER placebo regressions
- foreign-assembled placebo regressions
- canonical-sample and leave-one-country-out checks
- elasticity-interaction robustness
- price/markup decomposition tables
- alternative exposure definitions
- high-exposure visuals
- short literature-grounded interpretation

## Explicitly Excluded

- source-country fixed effects
- lagged-RER or timing-triplet specifications

## Build Dependency Order

1. `python cost_side/build_cost_side_panel.py`
2. `python cost_side/backfill_exchange_rates_2025.py` using cached IMF backfill values by default; add `--refresh` only to query IMF again
3. `Rscript cost_side/cost_reg_placebo.R`
4. `Rscript cost_side/cost_reg_robustness.R`
5. `Rscript cost_side/cost_reg_leave_one_country_out.R`
6. `Rscript cost_side/plot_cost_reduced_form.R`
7. `Rscript cost_side/cost_reg_price_markup_decomp.R`
8. `Rscript cost_side/cost_reg_alt_exposure.R`
9. `Rscript cost_side/plot_high_exposure_series.R`
10. `python cost_side/render_full_robustness_report_assets.py`
11. `xelatex` twice on `cost_side/docs/full_robustness_report.tex`
