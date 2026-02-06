# post_est

Entry points (main scripts):
- `get_elas_div.ipynb`
- `run_cf.ipynb`
- `make_graphs.ipynb`
- `plot_price_coef.ipynb`

Folder layout:
- `helpers/`: analysis helpers used by notebooks
  - `consumer_surplus.py`: manual CS calculations and switching metrics
  - `counterfactual_helpers.py`: counterfactual simulation helpers
  - `counterfactual_costs_prep.py`: prepare costs tables for CF runs
  - `counterfactual_profit_tables.py`: firm profit summary tables
  - `ev_tariff_metrics.py`: EV/tariff summary metrics
- `data/raw/`: raw inputs used by notebooks
- `data/derived/`: generated intermediate data (e.g., costs tables)
- `data/results/`: estimation outputs (pickles)
- `outputs/`: saved figures/tables from notebooks
- `redundant/`: older or unused notebooks, helpers, and data
