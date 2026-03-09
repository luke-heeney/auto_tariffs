# auto_tariffs

Replication repo for the EV BLP demand system, the cost-side exchange-rate regressions, and the post-estimation tariff/subsidy counterfactuals.

## Start here

The repo has three main workstreams:

- `estimation/blp_estimation/` fits the BLP demand model and writes result pickles.
- `cost_side/` estimates cost regressions and pass-through relationships used to motivate the parts-cost adjustment.
- `post_est/` loads an estimated model and runs counterfactual simulations, tables, and figures.

If you are new to the repo, read these in order:

1. `estimation/blp_estimation/README.md`
2. `cost_side/README.md`
3. `post_est/README.md`

## Recommended entry points

- BLP estimation:
  ```bash
  python estimation/blp_estimation/blp_run.py
  ```
- Cost-side elasticity regressions:
  ```bash
  Rscript cost_side/cost_reg_elas.R
  ```
- Counterfactual batch run:
  ```bash
  python post_est/run_cf_batch.py
  ```

## Repository layout

- `estimation/` — model estimation code, micro moments, and result pickles.
- `cost_side/` — R scripts for cost regressions and pass-through analysis.
- `post_est/` — counterfactual engine, configs, helper modules, and saved scenario outputs.
- `processed_data/` — processed inputs used elsewhere in the repo.

## Important files

- `post_est/run_cf_batch.py` — main scripted entry point for counterfactual simulations.
- `post_est/results_config.json` — default run configuration.
- `post_est/results_config_constant_market_by_market.json` — market-by-market solver with constant parts pass-through.
- `post_est/results_config_elasticity_interaction.json` — market-by-market solver with elasticity-dependent parts pass-through.
- `cost_side/cost_reg_elas.R` — primary cost-side regression script used to estimate elasticity interaction terms.

## Generated outputs

This repo contains many generated tables, figures, and saved outputs for reference. New runs write into `cost_side/outputs/`, `post_est/outputs/`, and `post_est/saved_outputs/`.

Generated outputs should be treated as build artifacts. Prefer committing code, configs, and documentation unless an output file is explicitly required.

## Data and large files

This repo includes data and outputs. Large files, including some counterfactual inputs, may be tracked with Git LFS.
