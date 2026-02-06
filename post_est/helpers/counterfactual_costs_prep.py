"""
counterfactual_costs_prep.py (minimal)

One job: build the clean objects your CF notebook expects, with as little code as possible.

Inputs you already have / read:
- product_data: your BLP product-market dataframe (has market_year, firm_ids, plant_country, etc.)
- vehicle_costs_csv: vehicle_costs_markups_characteristics.csv
  (product_ids with costs/markups already merged onto product_data fields)
- pc_panel_csv: pc_data_panel.csv (product_ids, year, pcUSCA_pct)

Outputs:
- product_data_clean: filtered to ONLY product_ids that exist in vehicle_costs_csv, with costs/markups attached
- rho_data: the panel (for your alias pc_panel_data = rho_data if you want)
- costs_df2: year-specific costs table used by counterfactual helpers, including pcUSCA_pct with US-only imputation
- diag: small counts of imputation sources
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def prepare_costs_df2_for_year(
    *,
    product_data: pd.DataFrame,
    vehicle_costs_csv: str,
    pc_panel_csv: str,
    year: int = 2024,
    id_col: str = "product_ids",
    market_year_col: str = "market_year",
    firm_col: str = "firm_ids",
    plant_col: str = "plant_country",
    cost_col: str = "costs",
    markup_col: str = "markups",
    rho_year_col: str = "year",
    share_col: str = "pcUSCA_pct",
    us_value: str = "United States",
):
    # Allow product_data to use clustering_ids instead of product_ids.
    pd0 = product_data.copy()
    if id_col not in pd0.columns and id_col == "product_ids" and "clustering_ids" in pd0.columns:
        pd0[id_col] = pd0["clustering_ids"].astype(str)

    # --- 1) Load costs/markups from merged vehicle costs file ---
    costs_df = pd.read_csv(vehicle_costs_csv, usecols=[id_col, cost_col, markup_col])
    costs_df[id_col] = costs_df[id_col].astype(str)
    costs_df = costs_df.drop_duplicates(subset=[id_col], keep="last")

    # --- 2) Filter product_data to costs universe and attach costs/markups cleanly ---
    pd0[id_col] = pd0[id_col].astype(str)
    drop_cols = [c for c in [cost_col, markup_col] if c in pd0.columns]
    if drop_cols:
        pd0 = pd0.drop(columns=drop_cols)

    product_data_clean = pd0.merge(costs_df, on=id_col, how="inner")

    # --- 3) Load pc panel (rho_data) and take year slice ---
    rho_data = pd.read_csv(pc_panel_csv, usecols=[id_col, rho_year_col, share_col])
    rho_data[id_col] = rho_data[id_col].astype(str)
    rho_y = rho_data.loc[rho_data[rho_year_col] == year, [id_col, share_col]].copy()
    rho_y[share_col] = pd.to_numeric(rho_y[share_col], errors="coerce")

    # --- 4) Build costs_df2 for the target year ---
    df = product_data_clean.loc[
        product_data_clean[market_year_col] == year,
        [id_col, firm_col, plant_col, market_year_col, cost_col]
    ].copy()
    df[cost_col] = pd.to_numeric(df[cost_col], errors="coerce")
    df = df.merge(rho_y, on=id_col, how="left")

    # --- 5) Carry forward pcUSCA_pct for any missing values (product_id, make_model, then firm) ---
    missing_any = df[share_col].isna()
    carry_forward_fills = 0
    carry_forward_make_model_fills = 0
    carry_forward_firm_fills = 0
    if missing_any.any():
        rho_all = rho_data[[id_col, rho_year_col, share_col]].copy()
        rho_all[id_col] = rho_all[id_col].astype(str)
        rho_all[rho_year_col] = pd.to_numeric(rho_all[rho_year_col], errors="coerce")
        rho_all[share_col] = pd.to_numeric(rho_all[share_col], errors="coerce")
        rho_all = rho_all.dropna(subset=[rho_year_col, share_col])
        rho_all = rho_all.loc[rho_all[rho_year_col] <= year]

        # 5a) Carry forward by exact product_id
        rho_all = rho_all.sort_values([id_col, rho_year_col])
        last_known = rho_all.groupby(id_col, dropna=False)[share_col].last()
        before = df[share_col].isna().sum()
        df.loc[missing_any, share_col] = df.loc[missing_any, id_col].map(last_known)
        after = df[share_col].isna().sum()
        carry_forward_fills = int(before - after)

        # 5b) Carry forward by make_model (strip year prefix)
        missing_any = df[share_col].isna()
        if missing_any.any():
            rho_all = rho_all.assign(make_model=rho_all[id_col].str.split("_", n=1).str[1])
            rho_mm = (rho_all.dropna(subset=["make_model"])
                              .sort_values(["make_model", rho_year_col])
                              .groupby("make_model", dropna=False)[share_col].last())
            df_make_model = df[id_col].str.split("_", n=1).str[1]
            before = df[share_col].isna().sum()
            df.loc[missing_any, share_col] = df_make_model.loc[missing_any].map(rho_mm)
            after = df[share_col].isna().sum()
            carry_forward_make_model_fills = int(before - after)

        # 5c) Carry forward by firm_id (use product_data to link firms across years)
        missing_any = df[share_col].isna()
        if missing_any.any() and firm_col in pd0.columns:
            pd_firm = pd0[[id_col, firm_col, market_year_col]].copy()
            pd_firm[id_col] = pd_firm[id_col].astype(str)
            pd_firm[market_year_col] = pd.to_numeric(pd_firm[market_year_col], errors="coerce")
            rho_firm = rho_data[[id_col, rho_year_col, share_col]].copy()
            rho_firm[id_col] = rho_firm[id_col].astype(str)
            rho_firm[rho_year_col] = pd.to_numeric(rho_firm[rho_year_col], errors="coerce")
            rho_firm[share_col] = pd.to_numeric(rho_firm[share_col], errors="coerce")
            rho_firm = rho_firm.dropna(subset=[rho_year_col, share_col])
            rho_firm = rho_firm.merge(
                pd_firm,
                left_on=[id_col, rho_year_col],
                right_on=[id_col, market_year_col],
                how="left",
            )
            rho_firm = rho_firm.dropna(subset=[firm_col])
            rho_firm = rho_firm.loc[rho_firm[rho_year_col] <= year]
            rho_firm = rho_firm.sort_values([firm_col, rho_year_col])
            firm_last = rho_firm.groupby(firm_col, dropna=False)[share_col].last()
            before = df[share_col].isna().sum()
            df.loc[missing_any, share_col] = df.loc[missing_any, firm_col].map(firm_last)
            after = df[share_col].isna().sum()
            carry_forward_firm_fills = int(before - after)

    # --- 6) Impute remaining missing pcUSCA_pct for US-assembled only (make_model mean in year, else firm mean) ---
    us_mask = df[plant_col].astype(str).eq(us_value)
    target = us_mask & df[share_col].isna()

    # make_model = everything after first underscore in product_ids ("YYYY_...")
    make_model_all = df[id_col].str.split("_", n=1).str[1]

    mm_map = (df.loc[us_mask, [share_col]].assign(make_model=make_model_all[us_mask])
              .dropna(subset=[share_col, "make_model"])
              .groupby("make_model")[share_col].mean())

    firm_map = (df.loc[us_mask, [firm_col, share_col]]
                .dropna(subset=[share_col, firm_col])
                .groupby(firm_col)[share_col].mean())

    fill_mm = make_model_all[target].map(mm_map)
    fill_fm = df.loc[target, firm_col].map(firm_map)
    fill = fill_mm.combine_first(fill_fm)

    df.loc[target, share_col] = fill

    # --- 7) Final fallback for any remaining missing (country-specific mean) ---
    remaining = df[share_col].isna()
    fallback_fills = 0
    if remaining.any():
        us_mean = df.loc[us_mask & df[share_col].notna(), share_col].mean()
        non_us_mean = df.loc[~us_mask & df[share_col].notna(), share_col].mean()
        overall_mean = df.loc[df[share_col].notna(), share_col].mean()
        fill_us = us_mean if pd.notna(us_mean) else overall_mean
        fill_non_us = non_us_mean if pd.notna(non_us_mean) else overall_mean
        before = df[share_col].isna().sum()
        df.loc[remaining & us_mask, share_col] = fill_us
        df.loc[remaining & ~us_mask, share_col] = fill_non_us
        after = df[share_col].isna().sum()
        fallback_fills = int(before - after)

    diag = {
        "carry_forward_fills": carry_forward_fills,
        "carry_forward_make_model_fills": carry_forward_make_model_fills,
        "carry_forward_firm_fills": carry_forward_firm_fills,
        "filled_from_make_model": int(fill_mm.notna().sum()),
        "filled_from_firm_mean": int((fill_mm.isna() & fill_fm.notna()).sum()),
        "remaining_missing_us_pcUSCA_pct": int((us_mask & df[share_col].isna()).sum()),
        "fallback_country_mean_fills": fallback_fills,
    }

    # --- 8) Finalize costs_df2 ---
    costs_df2 = df.drop_duplicates(subset=[id_col], keep="last").reset_index(drop=True)
    if share_col not in costs_df2.columns:
        costs_df2[share_col] = np.nan

    return product_data_clean, rho_data, costs_df2, diag
