# Downstream Assumptions Registry

This registry records outcome-relevant assumptions downstream of estimation and
where they are enforced.

| Assumption | Canonical source | Checked by |
|---|---|---|
| Fixed estimation input | `post_est/results_config.json: results_file` | `check_downstream_consistency.py` |
| Product and agent data | `post_est/results_config.json` | `check_downstream_consistency.py` |
| Canonical counterfactual config | `post_est/results_config.json` | root `make.sh`, `post_est/make.sh` |
| Counterfactual market year | `post_est/run_cf_batch.py: CS_MARKET_ID = 2024` | metadata check |
| Total market size | `post_est/run_cf_batch.py: TOTAL_MARKET_SIZE = 132_216_000 / 6` | metadata and unit checks |
| Price scale | `post_est/run_cf_batch.py: PRICE_SCALE_USD_PER_UNIT = 100_000` | metadata check |
| Income transform | `post_est/run_cf_batch.py: INCOME_TRANSFORM = "log_10k"` | metadata check when emitted |
| Baseline scenario | `post_est/run_cf_batch.py: baseline_label = "no tariff (no subsidy)"` | metadata and B0 zero checks |
| Scenario definitions | `post_est/helpers/counterfactual_reporting.py: default_scenario_specs` | scenario-index checks |
| Vehicle tariff | `post_est/run_cf_batch.py: VEHICLE_TARIFF = 0.25` | metadata and EV tariff table checks |
| Parts tariff | `post_est/run_cf_batch.py: PARTS_TARIFF = 0.25` | metadata and EV tariff table checks |
| Country-specific vehicle tariffs | `post_est/run_cf_batch.py: COUNTRY_TARIFFS` with canonical labels `United Kingdom`, `Japan`, `Korea` | metadata and country-normalization checks |
| Country parsing and aliases | `country_normalization.py`; applied in `cost_side/build_cost_side_panel.py`, post-estimation helpers, paper assets, and rebasing | `check_downstream_consistency.py`; `cost_side/outputs/country_parsing_summary.csv` |
| Parts pass-through | `post_est/results_config.json: parts_cost_adjustment.constant_pass_through = 0.5980` | config and metadata checks |
| Elasticity-interaction pass-through robustness | `post_est/results_config_elasticity_interaction.json` and `cost_side/outputs/cost_reg_elas_primary_spec_coeffs.csv` | path and output checks |
| 2025 exchange-rate backfill for future-RER placebos | cached IMF values in `cost_side/outputs/exchange_rate_2025_backfill_values.csv`; live refresh requires `--refresh` | cost-side output checks |
| Rebased reporting bundle | `post_est/rebase_saved_outputs_b0.py` metadata with `reporting_baseline_label` | bundle locator and paper manifest checks |
| Ownership mode | `post_est/results_config.json: ownership_mode = "owner"` | metadata check |
| Owner/pricer mapping | `post_est/data/raw/brand_owner_hq.xlsx` | config path and metadata checks |
| US-headquartered firm classification | `post_est/run_cf_batch.py: US_FIRMS` | graph-value and producer-surplus pipeline |
| Subsidy on/off scenarios | `default_scenario_specs(...).subsidy_zero` | scenario-index and EV table checks |
| EV definition for model outputs | `engine_type == "Electric"` in post-estimation figure builders | documented in slide/paper assets |
| Hybrid/PHEV treatment | Hybrids are distinct from `Electric`; PHEVs are not separately flagged in canonical outputs | manual review item if plug-in share is reported |

The constant `0.5980` is the rounded canonical domestic levels coefficient from
`cost_side/outputs/cost_reg_robustness_coefficients.csv`
(`canonical_domestic`, `levels_baseline`,
`ln_inv_rer_code1:pcOth1_pct1_lag1`). The cost-side elasticity-interaction
outputs are available and documented, but are not used by the default config
unless the canonical config is changed.
