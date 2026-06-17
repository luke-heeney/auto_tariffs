# Cost-Side Panel Build Decisions

This document records the sample-definition rules and construction choices for the exchange-rate identification robustness workstream.

## Raw Inputs

The scripted panel builder reads:

- `processed_data/auto_sourcing/percentage_data/*_manuf1.csv`
- `post_est/data/raw/blpUS0804.csv`
- `post_est/data/derived/vehicle_costs_markups_chars.csv`
- `processed_data/auto_sourcing/parts_imports_countries.csv`
- `processed_data/exchange_rates/exchange_rates.csv`

## Core Construction Rules

### Country normalization

Country parsing and country-specific tariff logic use the shared `country_normalization.py` helper. The cost-side builder applies this normalization before exchange-rate joins, and post-estimation code applies the same normalization before vehicle-tariff and tariff-revenue calculations.

Key decisions:

- `UK`, `GB`, `GBR`, and `Great Britain` are canonicalized to `United Kingdom`.
- `Korea` and `South Korea` are canonicalized to `Korea`.
- `China`, `CH`, and `CHN` are canonicalized to `China`.
- AALA-style `G` and observed `DE` raw source codes are treated as `Germany`.
- `A`, `AU`, and `AT` are treated as `Austria`.
- `CN` is treated as `Canada`, following the existing AALA-style source-code mapping and observed raw-file usage.
- `CZ` and `Czech Republic` are canonicalized to `Czechia`, matching the exchange-rate country name.

The panel builder writes two diagnostics in `cost_side/outputs/`: `country_parsing_raw_values.csv` and `country_parsing_summary.csv`. The downstream consistency checker fails if parsed country tokens are unmapped or if generated panels contain non-canonical labels such as `Great Britain` or `South Korea`.

### Product-year key

- `product_ids` is the raw vehicle key from the manufacturer files.
- `year` is the leading four-digit prefix in `product_ids`.
- `make_model` is the remainder of `product_ids` after stripping the year prefix.

### Duplicate-row collapse

The raw manufacturer files contain repeated rows for the same `product_ids`. The builder collapses these rows to one product-year record using the same high-level logic as the notebook:

- keep rows with any observed foreign-content slot if such rows exist,
- average `pcUSCA_pct` across the retained rows,
- aggregate foreign shares by country across all observed foreign slots,
- rank countries by average foreign share,
- fill `pcOth1_*` and `pcOth2_*` from that ranked list,
- carry forward modal metadata such as `assembly1`, `market_year`, `plant_country`, and `vehicle_type`.

This collapse is cross-row averaging within a product-year. It is not a time-series fill across years.

## Sample Flags

### `is_us_assembled`

`TRUE` when `plant_country == "United States"` after the BLP merge.

### `is_foreign_assembled`

`TRUE` when `plant_country` is observed and differs from `"United States"`.

### `source_country_switch_count`

For each `make_model`, this is the number of distinct observed `pcOth1_code1` values after the exchange-rate merge and after dropping rows with missing `rer_pcOth1_code1_n2015`.

### `source_country_stable`

`TRUE` when `source_country_switch_count <= 1`.

### `raw_row_count`

The number of raw manufacturer rows contributing to the collapsed `product_ids` record.

### `raw_rows_with_any_foreign_content`

The number of raw rows for the product-year with any non-missing foreign-content slot before collapse.

### `raw_primary_pair_count`

The number of raw rows for the product-year with both `pcOth1_code1` and `pcOth1_pct1` observed directly before collapse.

### `raw_primary_country_count`

The number of distinct directly observed raw `pcOth1_code1` countries among the rows for the product-year.

### `raw_primary_countries`

The pipe-separated list of directly observed raw primary countries for the product-year.

### `has_direct_primary_country`

`TRUE` when `raw_primary_country_count > 0`.

### `primary_source_country_consistent`

`TRUE` when `raw_primary_country_count <= 1`. This allows duplicate rows with different reported shares when the raw primary source country agrees.

### `canonical_exposure_ok`

`TRUE` when all of the following hold:

- `source_country_stable`,
- `has_direct_primary_country`,
- `primary_source_country_consistent`.

This is the only canonical domestic cost-side regression rule. It prevents collapsing across conflicting raw primary source countries, while allowing share averaging across duplicate rows with the same raw primary source country.

## Panel Variants

### `cost_side/cost_side_panel_all.csv`

All assembled vehicles that survive the panel build, cost merge, and exchange-rate merge, with flags attached.

### `cost_side/cost_side_panel.csv`

Canonical domestic regression panel: U.S.-assembled rows with `canonical_exposure_ok == TRUE`. This file is written as a slim analysis panel. It omits the audit fields, validation flags, `is_us_assembled`, `is_foreign_assembled`, and the raw `assembly1` label because these are either constant or only needed to verify construction decisions.

### `cost_side/cost_side_panel_foreign.csv`

Foreign-assembly subset with the same construction and flag columns.

## Interpretation Notes

- The current notebook does not show explicit forward-fill or backward-fill of `pcOth1_*` across years. The canonical rule therefore focuses on preventing cross-country duplicate-row collapse rather than claiming to undo a time-series imputation step that is not visibly present in the notebook.
- Duplicate product-year rows may be averaged when their directly observed primary source country agrees. Product-years with conflicting raw primary source countries are written to `cost_side/outputs/primary_source_country_conflicts.csv` and excluded from `cost_side/cost_side_panel.csv`.
- The foreign placebo uses the same exposure and exchange-rate construction as the domestic baseline. Its purpose is not to estimate a structural foreign cost pass-through parameter. It is a diagnostic check on whether the domestic identifying pattern is appearing in a sample where the paper’s claimed U.S.-assembled imported-parts channel should be much weaker.
- The levels future-RER placebo should attach `RER_{t+1}` directly from the current row's source country and `year + 1`. It should not require a next observed row for the same make-model.
- The first-difference future-RER placebo should define the future term directly as `\log(RER_{t+1}) - \log(RER_t)` on the current row, not as a lead of `\Delta \log(RER_t)` within make-model.
- The 2025 exchange-rate backfill now covers every source country used by the 2024 domestic future-placebo rows. The documented values and country mapping are in `cost_side/outputs/exchange_rate_2025_backfill_summary.md`, with cached values in `cost_side/outputs/exchange_rate_2025_backfill_values.csv`. The default replication path uses those cached values; `--refresh` is required to query IMF again.
