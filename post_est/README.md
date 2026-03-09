# post_est

This directory contains the post-estimation counterfactual pipeline. It loads an estimated BLP model, prepares cost inputs, solves tariff/subsidy scenarios, and writes summary tables and figures.

## Main entry point

The recommended scripted entry point is:

```bash
python post_est/run_cf_batch.py
```

`run_cf_batch.py` is the best place to start if you want a reproducible batch run instead of a notebook workflow.

## What `run_cf_batch.py` does

1. reads a results config JSON,
2. loads the BLP results pickle and the matching product/agent data,
3. prepares a 2024 cost table with parts-content shares and optional product elasticities,
4. runs the default tariff/subsidy scenario set,
5. writes summary tables, per-scenario CSVs, and figures into `post_est/saved_outputs/`, and
6. records metadata so an identical run can reuse prior saved outputs.

## Key configs

- `results_config.json` — default config.
- `results_config_constant_market_by_market.json` — alternate config using the market-by-market solver with constant parts pass-through.
- `results_config_elasticity_interaction.json` — alternate config using the market-by-market solver with elasticity-dependent parts pass-through.

Each config selects:

- the estimation results pickle,
- product and agent input files,
- ownership mapping behavior, and
- `parts_cost_adjustment`, which controls whether parts-cost pass-through is constant or varies with own-price elasticity.

You can choose a non-default config with:

```bash
RESULTS_CONFIG_PATH=post_est/results_config_elasticity_interaction.json python post_est/run_cf_batch.py
```

## Folder layout

- `helpers/` — reusable analysis helpers.
  - `counterfactual_costs_prep.py` — merges costs, parts-content shares, and optional elasticities.
  - `counterfactual_helpers.py` — cost construction, simulation wrappers, and scenario-level output tables.
  - `counterfactual_reporting.py` — scenario definitions and report/table assembly.
  - `consumer_surplus.py` — consumer-surplus calculations.
  - `counterfactual_profit_tables.py` — firm/owner profit summaries.
  - `ev_tariff_metrics.py` — EV-share and tariff-revenue summary tables.
  - `ownership.py` — owner/pricer mapping utilities.
- `data/raw/` — raw product, agent, ownership, and panel inputs.
- `data/derived/` — generated intermediates such as `vehicle_costs_markups_chars.csv` and `product_year_elasticities.csv`.
- `data/results/` — saved estimation outputs (`.pkl`).
- `outputs/` — lightweight rendered tables and figures.
- `saved_outputs/` — full counterfactual run directories with per-scenario outputs and figures.

## Solver modes

Two solver paths are currently supported:

- `unified` — use the original unified counterfactual routine.
- `market_by_market` — solve scenarios through the market-by-market path and rebuild summary outputs from those results.

The solver mode is controlled by `counterfactual_solver_mode` in the results config.

## Generated outputs

Typical run artifacts include:

- `summary_tbl_all.csv.gz`
- `ev_tariff_tbl.csv.gz`
- per-scenario product, firm, owner, market-surplus, and overall-surplus tables
- figures under `saved_outputs/<run_id>/figures/`

Treat `outputs/` and `saved_outputs/` as generated artifacts rather than hand-edited source files.
