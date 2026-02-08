# BLP Estimation (Dec17)

This directory contains the current BLP estimation pipeline and the minimum data needed to run it. The main entry point is `blp_run.py` and the default SLURM runner is `batch_blp.sh`.

## Quick start

- Local:
  - Run: `python blp_run.py`
- SLURM:
  - Submit: `sbatch batch_blp.sh`

## Main scripts

- `blp_run.py`: main estimation script; builds product + agent data, constructs micro moments, and solves the model.
- `data_prep.py`: product data cleaning and feature construction.
- `agent_prep.py`: agent data loading and preprocessing.
- `micromoments/`: micro-moment builders.
- `batch_blp.sh`: SLURM batch script to run `blp_run.py` on the server.

## Data layout

Active inputs in `data/`:
- `blp_with_subsidies.csv`: baseline product data with subsidies.
- `blp_with_45W_subsidies_scale1p0.csv`: product data with 45W subsidies.
- `agent_incomes_400perdivision.csv`: agent microdata with division labels.
- `prob_purchase_given_division.csv`, `prob_EV_given_division.csv`, `prob_vehicle_type_given_division.csv`: division-level targets for micro moments.

Archived inputs live in `data/archive/` (older versions or unused sources).

Outputs:
- `prints/`: job stdout/stderr from SLURM.
- `pickles/`: model result pickles.

## Key settings in `blp_run.py`

Top-of-file configuration:

- `shares_scaling`: `None`, `'halve'`, or `'double'` (rescales product shares).
- `include_45W_subsidies`: `0/1` toggle that switches product data between:
  - `data/blp_with_subsidies.csv` (0)
  - `data/blp_with_45W_subsidies_scale1p0.csv` (1)
  This also tags output filenames with `subsidy` vs `45W`.

Flags (0/1) in `flag_values`:
- `include_us`, `include_eu`: brand-region effects.
- `include_hyb_in_mpg`: whether hybrids are included in mpg effects.
- `include_ev_in_lux`: whether EV brands count as luxury.
- `include_ev_in_us`: whether EV brands count as US brands.
- `include_mpg`, `include_size`, `include_hp`: random-coefficient inclusion.
- `include_prob_moments`: add income-based purchase-probability moments.
- `include_GH_instruments`: toggle GH instruments (if used in downstream code).
- `include_old_models`: include non-current-year models.
- `include_price_mpg_int`: allow income×mpg interactions in Pi.
- `include_outside_rc`: allow random coefficient on outside option.
- `first_log`: log transform on the first random draw.
- `include_div_purchase_moments`, `include_div_ev_moments`, `include_div_type_moments`:
  division-level micro moments.
- `use_halton`: Halton draws for unobserved heterogeneity.
- `include_price`: random coefficient on price.

## Notes

- `agent_prep.py` expects `division` in the agent CSV.
- `blp_run.py` writes intermediate `product_data_*.csv` and `agent_data_*.csv` in the repo root, tagged by subsidy mode.
