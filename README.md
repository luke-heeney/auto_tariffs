# auto_tariffs

Replication repo for the EV BLP demand system and counterfactual tariff/subsidy analysis.

## Repository layout
- `estimation/` — BLP estimation code and inputs.
- `post_est/` — counterfactuals, output tables, and figures.
- `cost_side/` — cost-side regressions.
- `processed_data/` — processed data inputs used by scripts.

## Key files
- `post_est/run_cf_batch.py` — runs counterfactuals and regenerates figures/tables.
- `post_est/results_config.json` — selects results and data inputs.
- `post_est/saved_outputs/` — saved counterfactual outputs and figures.

## Running counterfactuals
1. Edit `post_est/results_config.json` to point to the results file and data paths.
2. Run:
   ```bash
   python post_est/run_cf_batch.py
   ```

This script reuses existing counterfactual outputs for the same results file and regenerates figures only, unless outputs are missing.

## Data and large files
This repo includes data and outputs. Large files (e.g., `post_est/data/raw/agent_data_cf.csv`) are tracked with Git LFS.

## Notes
- The “no subsidy” counterfactual sets the `subsidy` column in product data to 0 before simulation.
- Figures are saved as PNGs under `post_est/saved_outputs/<results_stem>/figures/`.

