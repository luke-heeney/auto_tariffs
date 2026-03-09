# cost_side

This directory contains the cost-side regressions used to relate exchange rates, imported-parts exposure, and vehicle costs.

## What is here

- `cost_reg.R` — baseline cost regression script.
- `cost_reg_elas.R` — elasticity-interaction regression script. This is the main script if you want the pass-through coefficients used by `post_est/`.
- `cost_reg_vehicle_type.R` — vehicle-type splits.
- `plot_implied_pass_through.R` — plots implied pass-through against elasticity.
- `run_cost_side.ipynb` — notebook version of the cost-side workflow.
- `cost_side_panel.csv` / `cost_side_panel_dropped.csv` — panel inputs for the regressions.
- `outputs/` — generated tables, figures, and CSVs.

## Recommended workflow

Run the elasticity-interaction regression from the repository root:

```bash
Rscript cost_side/cost_reg_elas.R
```

That script:

1. loads the cost-side panel,
2. merges in product-year elasticities from `post_est/data/derived/product_year_elasticities.csv`,
3. estimates levels and first-difference specifications for the domestic-assembly sample,
4. writes LaTeX tables and diagnostics into `cost_side/outputs/`, and
5. exports the primary-spec coefficients to `cost_side/outputs/cost_reg_elas_primary_spec_coeffs.csv`.

## Why the coefficient CSV matters

`post_est/results_config*.json` can point to `cost_side/outputs/cost_reg_elas_primary_spec_coeffs.csv` when the counterfactuals use `parts_cost_adjustment.mode = "elasticity_interaction"`.

That file gives the intercept and elasticity-interaction slope used to convert a product's own-price elasticity into a parts-cost pass-through rate during the tariff simulations.

## Expected outputs

The most useful generated files are:

- `outputs/cost_reg_elas_merge_diagnostics.csv`
- `outputs/cost_reg_elas_levels_table.tex`
- `outputs/cost_reg_elas_fd_table.tex`
- `outputs/cost_reg_elas_primary_spec_coeffs.csv`

Treat files under `outputs/` as generated artifacts rather than hand-edited source files.
