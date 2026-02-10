"""
counterfactual_helpers.py

Lean counterfactual helpers, assuming fixed column schemas:

costs_df2 contains at least:
  ['product_ids','firm_ids','plant_country','market_year','costs','pcUSCA_pct']

product_data contains at least:
  ['market_ids','product_ids','firm_ids','market_year','prices','plant_country']

No "detect column" logic, no extra validation. NaN pcUSCA_pct => ignored for parts tariff (inflator=1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyblp
from helpers.consumer_surplus import cs_manual_all_markets
from helpers.ownership import add_owner_ids, build_owner_ownership, attach_ownership_columns


def build_counterfactual_costs(
    costs_df2: pd.DataFrame,
    *,
    parts_tariff: float = 0.0,
    vehicle_tariff: float = 0.0,
    country_tariffs: dict[str, float] | None = None,
    parts_pass_through: float = 0.715,
    vehicle_pass_through: float = 0.682,
    us_value: str = "United States",
    id_col: str = "product_ids",
    plant_col: str = "plant_country",
    cost_col: str = "costs",
    share_col: str = "pcUSCA_pct",
):
    if id_col not in costs_df2.columns and "clustering_ids" in costs_df2.columns:
        costs_df2 = costs_df2.copy()
        costs_df2[id_col] = costs_df2["clustering_ids"].astype(str)

    df = costs_df2[[id_col, "firm_ids", plant_col, "market_year", cost_col, share_col]].copy()
    df[id_col] = df[id_col].astype(str)

    us = df[plant_col].astype(str).eq(us_value)
    base_cost = pd.to_numeric(df[cost_col], errors="coerce").to_numpy(dtype=float)

    infl = np.ones(len(df), dtype=float)

    if parts_tariff:
        s = pd.to_numeric(df[share_col], errors="coerce").to_numpy(dtype=float)
        imported_share = np.clip(1.0 - s, 0.0, 1.0)
        infl_us = 1.0 + imported_share * (parts_pass_through * parts_tariff)
        infl_us = np.where(np.isfinite(infl_us), infl_us, 1.0)  # ignore NaNs
        infl = np.where(us, infl_us, infl)

    if vehicle_tariff or country_tariffs:
        if country_tariffs:
            country = df[plant_col].astype(str).str.strip()
            rates = country.map(country_tariffs).fillna(vehicle_tariff).to_numpy(dtype=float)
            infl_foreign = 1.0 + vehicle_pass_through * rates
            infl = np.where(~us, infl_foreign, infl)
        else:
            infl_foreign = 1.0 + vehicle_pass_through * vehicle_tariff
            infl = np.where(~us, infl_foreign, infl)

    df["cost_inflator_cf"] = infl
    df["cost_cf"] = base_cost * infl
    return df, "cost_cf"


def _get_results_id_vector(results):
    products = results.problem.products
    if hasattr(products, "clustering_ids"):
        return np.asarray(products.clustering_ids).reshape(-1)
    return np.asarray(products.product_ids).reshape(-1)


def align_product_data_for_sim(
    results,
    product_data: pd.DataFrame,
    *,
    id_col: str = "product_ids",
    market_col: str = "market_ids",
):
    df = product_data.copy()
    if id_col not in df.columns and "clustering_ids" in df.columns:
        df[id_col] = df["clustering_ids"].astype(str)

    order_df = pd.DataFrame({
        id_col: _get_results_id_vector(results),
        market_col: np.asarray(results.problem.products.market_ids).reshape(-1),
    })

    df = order_df.merge(
        df,
        on=[id_col, market_col],
        how="left",
        validate="one_to_one",
    )

    levels = sorted(set(int(x) for x in np.asarray(order_df[market_col]).reshape(-1)))
    df[market_col] = df[market_col].astype(int)
    df[market_col] = pd.Categorical(df[market_col], categories=levels)
    return df


def build_costs_vector_from_vehicle_costs(
    results,
    vehicle_costs_df: pd.DataFrame,
    *,
    id_col: str = "product_ids",
    market_col: str = "market_ids",
    cost_col: str = "costs",
):
    df = vehicle_costs_df.copy()
    if id_col not in df.columns and "clustering_ids" in df.columns:
        df[id_col] = df["clustering_ids"].astype(str)

    order_df = pd.DataFrame({
        id_col: _get_results_id_vector(results),
        market_col: np.asarray(results.problem.products.market_ids).reshape(-1),
    })

    order_df[id_col] = order_df[id_col].astype(str)
    order_df[market_col] = pd.to_numeric(order_df[market_col], errors="coerce")

    df[id_col] = df[id_col].astype(str)
    df[market_col] = pd.to_numeric(df[market_col], errors="coerce")
    df = df[[id_col, market_col, cost_col]].drop_duplicates([id_col, market_col], keep="last")

    merged = order_df.merge(df, on=[id_col, market_col], how="left", validate="one_to_one")
    if merged[cost_col].isna().any():
        missing = merged.loc[merged[cost_col].isna(), [id_col, market_col]].head(5)
        raise ValueError(
            "Missing costs for some products. Ensure vehicle_costs_markups_chars.csv "
            f"matches the results file. Sample missing rows:\n{missing}"
        )

    return pd.to_numeric(merged[cost_col], errors="coerce").to_numpy(dtype=float)


def build_noabs_formulations(results, *, firm_col: str = "firm_ids"):
    x1_formula = results.problem.product_formulations[0]._formula
    fe_term = f"C({firm_col})"
    if fe_term not in x1_formula:
        x1_formula_noabs = f"{x1_formula} + {fe_term}"
    else:
        x1_formula_noabs = x1_formula
    X1_formulation_noabs = pyblp.Formulation(x1_formula_noabs)
    X2_formulation = results.problem.product_formulations[1]
    return X1_formulation_noabs, X2_formulation


def compute_beta_full(results, X1_formulation_noabs, aligned_product_data):
    X1_full, *_ = X1_formulation_noabs._build_matrix(aligned_product_data)
    delta0_full = np.asarray(results.delta).reshape(-1)
    xi_results_full = np.asarray(results.xi).reshape(-1)
    beta_full = np.linalg.lstsq(X1_full, delta0_full - xi_results_full, rcond=None)[0].reshape(-1, 1)
    return beta_full


def run_simulation(
    results,
    product_data: pd.DataFrame,
    *,
    X1_formulation_noabs,
    X2_formulation,
    beta_full,
    xi_full,
    agent_data,
    costs=None,
    fixed_prices=None,
    id_col: str = "product_ids",
    market_col: str = "market_ids",
    firm_col: str = "firm_ids",
    ownership_mode: str = "firm",
    owner_map: dict[str, str] | None = None,
    pricer_map: dict[str, str] | None = None,
    allow_unmapped_brands: bool = False,
):
    df = align_product_data_for_sim(
        results,
        product_data,
        id_col=id_col,
        market_col=market_col,
    )
    if ownership_mode == "owner":
        if pricer_map is None:
            raise ValueError("pricer_map is required when ownership_mode='owner'.")
        df = add_owner_ids(
            df,
            pricer_map,
            firm_col=firm_col,
            owner_col="pricer_ids",
            allow_unmapped=allow_unmapped_brands,
        )
        ownership = build_owner_ownership(df, owner_col="pricer_ids", market_col=market_col)
        df = attach_ownership_columns(df, ownership)
        if owner_map is not None:
            df = add_owner_ids(
                df,
                owner_map,
                firm_col=firm_col,
                owner_col="owner_ids",
                allow_unmapped=allow_unmapped_brands,
            )

    sim = pyblp.Simulation(
        product_formulations=(X1_formulation_noabs, X2_formulation),
        product_data=df,
        beta=beta_full,
        sigma=results.sigma,
        pi=results.pi,
        rho=results.rho,
        gamma=results.gamma,
        agent_formulation=results.problem.agent_formulation,
        agent_data=agent_data,
        xi=xi_full,
    )

    if costs is None:
        raise ValueError("costs must be provided (generate via get_elas_div and vehicle_costs_markups_chars.csv).")
    costs = np.asarray(costs).reshape(-1)

    if fixed_prices is not None:
        fixed_prices = np.asarray(fixed_prices).reshape(-1)
        sim_results = sim.replace_endogenous(
            costs=costs,
            prices=fixed_prices,
            iteration=pyblp.Iteration("return"),
        )
    else:
        sim_results = sim.replace_endogenous(costs=costs)

    return sim_results, df


def _year_mask(df, *, year: int, market_year_col: str = "market_year", market_col: str = "market_ids"):
    if market_year_col in df.columns:
        return df[market_year_col].eq(year)
    return df[market_col].eq(year)


def run_unified_counterfactual(
    results,
    product_data: pd.DataFrame,
    costs_df2: pd.DataFrame,
    *,
    agent_data,
    costs_full,
    ownership_mode: str = "firm",
    owner_map: dict[str, str] | None = None,
    pricer_map: dict[str, str] | None = None,
    allow_unmapped_brands: bool = False,
    year: int = 2024,
    market_col: str = "market_ids",
    id_col: str = "product_ids",
    firm_col: str = "firm_ids",
    plant_col: str = "plant_country",
    price_col: str = "prices",
    parts_tariff: float = 0.0,
    vehicle_tariff: float = 0.0,
    country_tariffs: dict[str, float] | None = None,
    parts_pass_through: float = 0.715,
    vehicle_pass_through: float = 0.682,
    subsidy_zero: bool = False,
    baseline_fixed_prices: bool = True,
    price_x2_index: int | None = None,
    beta_price_index: int | None = None,
    gamma: float = 0.0,
    total_market_size: float = 132_000_000 / 6,
    price_scale_usd_per_unit: float = 100_000.0,
):
    if id_col not in product_data.columns and "clustering_ids" in product_data.columns:
        product_data = product_data.copy()
        product_data[id_col] = product_data["clustering_ids"].astype(str)

    if price_x2_index is None or beta_price_index is None:
        raise ValueError("price_x2_index and beta_price_index are required to compute manual CS.")

    X1_formulation_noabs, X2_formulation = build_noabs_formulations(results, firm_col=firm_col)
    aligned_base = align_product_data_for_sim(
        results,
        product_data,
        id_col=id_col,
        market_col=market_col,
    )
    beta_full = compute_beta_full(results, X1_formulation_noabs, aligned_base)
    xi_full = np.asarray(results.xi).reshape(-1, 1)

    base_prices = aligned_base[price_col].to_numpy()
    if costs_full is None:
        raise ValueError("costs_full is required; generate via get_elas_div and load from vehicle_costs_markups_chars.csv.")
    costs_full = np.asarray(costs_full).reshape(-1)
    if costs_full.size != len(results.problem.products.market_ids):
        raise ValueError("costs_full length does not match results products.")
    sim_base, aligned_base = run_simulation(
        results,
        product_data,
        X1_formulation_noabs=X1_formulation_noabs,
        X2_formulation=X2_formulation,
        beta_full=beta_full,
        xi_full=xi_full,
        agent_data=agent_data,
        costs=costs_full,
        fixed_prices=base_prices if baseline_fixed_prices else None,
        id_col=id_col,
        market_col=market_col,
        firm_col=firm_col,
        ownership_mode=ownership_mode,
        owner_map=owner_map,
        pricer_map=pricer_map,
        allow_unmapped_brands=allow_unmapped_brands,
    )
    p0_full = np.asarray(sim_base.product_data["prices"]).reshape(-1)
    s0_full = np.asarray(sim_base.product_data["shares"]).reshape(-1)
    c0_full = np.asarray(sim_base.costs).reshape(-1)

    product_cf = product_data.copy()
    if subsidy_zero and "subsidy" in product_cf.columns:
        product_cf["subsidy"] = 0.0

    aligned_cf = align_product_data_for_sim(
        results,
        product_cf,
        id_col=id_col,
        market_col=market_col,
    )

    costs_full = np.asarray(costs_full).reshape(-1)
    apply_tariff = bool(parts_tariff or vehicle_tariff or (country_tariffs is not None))
    if apply_tariff:
        cf_costs_df, cf_cost_col = build_counterfactual_costs(
            costs_df2,
            parts_tariff=parts_tariff,
            vehicle_tariff=vehicle_tariff,
            country_tariffs=country_tariffs,
            parts_pass_through=parts_pass_through,
            vehicle_pass_through=vehicle_pass_through,
            id_col=id_col,
        )
        cf_cost_map = cf_costs_df.set_index(id_col)[cf_cost_col]

        year_mask = _year_mask(aligned_cf, year=year, market_col=market_col)
        ids_year = aligned_cf.loc[year_mask, id_col].astype(str)
        cf_costs_year = pd.to_numeric(cf_cost_map.reindex(ids_year), errors="coerce").to_numpy(dtype=float)
        cf_costs_year = np.where(np.isfinite(cf_costs_year), cf_costs_year, costs_full[year_mask])
        costs_full = costs_full.copy()
        costs_full[year_mask.to_numpy()] = cf_costs_year
    else:
        cf_costs_df = None
        cf_cost_col = None

    sim_cf, aligned_cf = run_simulation(
        results,
        product_cf,
        X1_formulation_noabs=X1_formulation_noabs,
        X2_formulation=X2_formulation,
        beta_full=beta_full,
        xi_full=xi_full,
        agent_data=agent_data,
        costs=costs_full,
        fixed_prices=None,
        id_col=id_col,
        market_col=market_col,
        firm_col=firm_col,
        ownership_mode=ownership_mode,
        owner_map=owner_map,
        pricer_map=pricer_map,
        allow_unmapped_brands=allow_unmapped_brands,
    )

    year_markets = aligned_base.loc[aligned_base["market_year"] == year, market_col].unique() \
        if "market_year" in aligned_base.columns else aligned_base.loc[aligned_base[market_col] == year, market_col].unique()

    prod_blocks = []
    for mid in year_markets:
        m_idx = aligned_base[market_col].eq(mid)
        ids_m = aligned_base.loc[m_idx, id_col].astype(str).to_numpy()

        p0 = p0_full[m_idx.to_numpy()]
        s0 = s0_full[m_idx.to_numpy()]
        c0 = c0_full[m_idx.to_numpy()]

        sub0 = None
        if "subsidy" in aligned_base.columns:
            sub0 = pd.to_numeric(aligned_base.loc[m_idx, "subsidy"], errors="coerce").to_numpy(dtype=float)
        sub_cf = None
        if "subsidy" in aligned_cf.columns:
            sub_cf = pd.to_numeric(aligned_cf.loc[m_idx, "subsidy"], errors="coerce").to_numpy(dtype=float)

        cf_idx = aligned_cf[market_col].eq(mid)
        p_cf = np.asarray(sim_cf.product_data["prices"]).reshape(-1)[cf_idx]
        s_cf = np.asarray(sim_cf.product_data["shares"]).reshape(-1)[cf_idx]
        c_cf = costs_full[cf_idx.to_numpy()]

        mu0 = p0 - c0
        mu_cf = p_cf - c_cf
        pi0 = mu0 * s0
        pi_cf = mu_cf * s_cf

        base_cols = [market_col, id_col, firm_col, plant_col]
        if "owner_ids" in aligned_base.columns:
            base_cols.append("owner_ids")
        block = aligned_base.loc[m_idx, base_cols].copy()
        block[id_col] = block[id_col].astype(str)

        block["p0"] = p0
        block["p_cf"] = p_cf
        block["dp"] = p_cf - p0
        block["dp_pct"] = 100.0 * np.where(p0 != 0, block["dp"] / p0, np.nan)

        if sub0 is not None:
            block["subsidy0"] = sub0
            block["p0_net"] = p0 - sub0
        if sub_cf is not None:
            block["subsidy_cf"] = sub_cf
            block["p_cf_net"] = p_cf - sub_cf

        block["c0"] = c0
        block["c_cf"] = c_cf
        block["dc"] = c_cf - c0
        block["dc_pct"] = 100.0 * np.where(c0 != 0, block["dc"] / c0, np.nan)

        block["s0"] = s0
        block["s_cf"] = s_cf
        block["ds"] = s_cf - s0
        block["ds_pct"] = 100.0 * np.where(s0 != 0, block["ds"] / s0, np.nan)

        block["mu0"] = mu0
        block["mu_cf"] = mu_cf
        block["dmu"] = mu_cf - mu0

        margin0 = 100.0 * np.where(p0 != 0, mu0 / p0, np.nan)
        margin_cf = 100.0 * np.where(p_cf != 0, mu_cf / p_cf, np.nan)
        block["margin0_pct"] = margin0
        block["margin_cf_pct"] = margin_cf
        block["dmargin_pct"] = margin_cf - margin0

        block["pi0"] = pi0
        block["pi_cf"] = pi_cf
        block["dpi"] = pi_cf - pi0

        prod_blocks.append(block)

    product_table = pd.concat(prod_blocks, ignore_index=True)
    market_table = cs_manual_all_markets(
        results,
        product_table,
        price_x2_index=price_x2_index,
        beta_price_index=beta_price_index,
        gamma=gamma,
        market_col=market_col,
        id_col=id_col,
    )

    def _build_group_table(group_col: str) -> pd.DataFrame:
        rows = []
        for firm, d in product_table.groupby(group_col, dropna=False):
            w0 = d["s0"].to_numpy(dtype=float)
            wcf = d["s_cf"].to_numpy(dtype=float)

            def wavg(x, w):
                den = float(np.nansum(w))
                return np.nan if den <= 0 else float(np.nansum(x.to_numpy(dtype=float) * w) / den)

            r = {
                group_col: firm,
                "share0_total": float(np.nansum(w0)),
                "share_cf_total": float(np.nansum(wcf)),
                "p0_sw": wavg(d["p0"], w0),
                "p_cf_sw": wavg(d["p_cf"], wcf),
                "c0_sw": wavg(d["c0"], w0),
                "c_cf_sw": wavg(d["c_cf"], wcf),
                "mu0_sw": wavg(d["mu0"], w0),
                "mu_cf_sw": wavg(d["mu_cf"], wcf),
                "pi0_percap_total": float(np.nansum(d["pi0"].to_numpy(dtype=float))),
                "pi_cf_percap_total": float(np.nansum(d["pi_cf"].to_numpy(dtype=float))),
            }
            r["dp_sw"] = r["p_cf_sw"] - r["p0_sw"]
            r["dc_sw"] = r["c_cf_sw"] - r["c0_sw"]
            r["dmu_sw"] = r["mu_cf_sw"] - r["mu0_sw"]
            r["dpi_percap_total"] = r["pi_cf_percap_total"] - r["pi0_percap_total"]
            rows.append(r)
        tbl = pd.DataFrame(rows)
        factor_musd = float(total_market_size) * (price_scale_usd_per_unit / 1_000_000.0)
        tbl["pi0_millions_usd"] = tbl["pi0_percap_total"] * factor_musd
        tbl["pi_cf_millions_usd"] = tbl["pi_cf_percap_total"] * factor_musd
        tbl["dpi_millions_usd"] = tbl["dpi_percap_total"] * factor_musd
        return tbl.sort_values("dpi_millions_usd", ascending=False).reset_index(drop=True)

    firm_table = _build_group_table(firm_col)
    owner_table = _build_group_table("owner_ids") if "owner_ids" in product_table.columns else None

    factor_musd = float(total_market_size) * (price_scale_usd_per_unit / 1_000_000.0)

    market_table["CS0_millions_usd"] = market_table["CS0"] * factor_musd
    market_table["CS_cf_millions_usd"] = market_table["CS_cf"] * factor_musd
    market_table["dCS_millions_usd"] = market_table["dCS"] * factor_musd

    overall = pd.DataFrame([{
        "year": year,
        "total_firm_surplus_change_millions_usd": float(firm_table["dpi_millions_usd"].sum()),
        "total_consumer_surplus_change_millions_usd": float(market_table["dCS_millions_usd"].sum()),
        "assumptions_total_market_size": float(total_market_size),
        "assumptions_price_scale_usd_per_unit": float(price_scale_usd_per_unit),
    }])

    return {
        "product_table": product_table,
        "firm_table": firm_table,
        "owner_table": owner_table,
        "market_surplus_table": market_table,
        "overall_surplus": overall,
        "cf_costs_df": cf_costs_df,
        "cf_cost_col": cf_cost_col,
    }


def run_cf(
    results,
    product_data: pd.DataFrame,
    costs_df2: pd.DataFrame,
    *,
    costs_full,
    year: int = 2024,
    market_col: str = "market_ids",
    id_col: str = "product_ids",
    firm_col: str = "firm_ids",
    parts_tariff: float = 0.0,
    vehicle_tariff: float = 0.0,
    country_tariffs: dict[str, float] | None = None,
    parts_pass_through: float = 0.715,
    vehicle_pass_through: float = 0.682,
    iter_cfg=None,
    ownership_mode: str = "firm",
    owner_map: dict[str, str] | None = None,
    pricer_map: dict[str, str] | None = None,
    allow_unmapped_brands: bool = False,
):
    if costs_full is None:
        raise ValueError("costs_full is required; generate via get_elas_div and load from vehicle_costs_markups_chars.csv.")
    costs_full = np.asarray(costs_full).reshape(-1)

    # Allow product_data to use clustering_ids instead of product_ids.
    if id_col not in product_data.columns and "clustering_ids" in product_data.columns:
        product_data = product_data.copy()
        product_data[id_col] = product_data["clustering_ids"].astype(str)

    aligned_pd = align_product_data_for_sim(results, product_data, id_col=id_col, market_col=market_col)
    if ownership_mode == "owner":
        if pricer_map is None:
            raise ValueError("pricer_map is required when ownership_mode='owner'.")
        aligned_pd = add_owner_ids(
            aligned_pd,
            pricer_map,
            firm_col=firm_col,
            owner_col="pricer_ids",
            allow_unmapped=allow_unmapped_brands,
        )

    cf_costs_df, cf_cost_col = build_counterfactual_costs(
        costs_df2,
        parts_tariff=parts_tariff,
        vehicle_tariff=vehicle_tariff,
        country_tariffs=country_tariffs,
        parts_pass_through=parts_pass_through,
        vehicle_pass_through=vehicle_pass_through,
        id_col=id_col,
    )
    cf_cost_map = cf_costs_df.set_index(id_col)[cf_cost_col]

    year_markets = aligned_pd.loc[aligned_pd["market_year"] == year, market_col].unique()

    blocks = []
    for mid in year_markets:
        m_idx = aligned_pd[market_col].eq(mid)
        ids_m = aligned_pd.loc[m_idx, id_col].astype(str).to_numpy()

        market_ids_full = np.asarray(results.problem.products.market_ids).reshape(-1)
        c0 = costs_full[market_ids_full == mid]

        cf_m = pd.to_numeric(cf_cost_map.reindex(ids_m), errors="coerce").to_numpy(dtype=float)
        c_m = np.where(np.isfinite(cf_m), cf_m, c0)

        ownership = None
        if ownership_mode == "owner":
            m_owner = aligned_pd.loc[m_idx, [market_col, "pricer_ids"]].copy()
            m_owner = m_owner.rename(columns={"pricer_ids": "firm_ids"})
            ownership = np.asarray(pyblp.build_ownership(m_owner), dtype=float)

        p_m = np.asarray(
            results.compute_prices(costs=c_m, market_id=mid, ownership=ownership, iteration=iter_cfg),
            dtype=float
        ).reshape(-1)

        s_m = np.asarray(results.compute_shares(prices=p_m, market_id=mid), dtype=float).reshape(-1)

        blocks.append(pd.DataFrame({market_col: mid, id_col: ids_m, "p_cf": p_m, "s_cf": s_m}))

    return pd.concat(blocks, ignore_index=True), cf_costs_df, cf_cost_col


def run_cf_and_summarize(
    results,
    product_data: pd.DataFrame,
    costs_df2: pd.DataFrame,
    *,
    costs_full,
    year: int = 2024,
    market_col: str = "market_ids",
    id_col: str = "product_ids",
    firm_col: str = "firm_ids",
    plant_col: str = "plant_country",
    price_col: str = "prices",
    parts_tariff: float = 0.0,
    vehicle_tariff: float = 0.0,
    country_tariffs: dict[str, float] | None = None,
    parts_pass_through: float = 0.715,
    vehicle_pass_through: float = 0.682,
    price_x2_index: int | None = None,
    beta_price_index: int | None = None,
    gamma: float = 0.0,
    iter_cfg=None,
    total_market_size: float = 132_000_000 / 6,
    price_scale_usd_per_unit: float = 100_000.0,
    ownership_mode: str = "firm",
    owner_map: dict[str, str] | None = None,
    pricer_map: dict[str, str] | None = None,
    allow_unmapped_brands: bool = False,
):
    if costs_full is None:
        raise ValueError("costs_full is required; generate via get_elas_div and load from vehicle_costs_markups_chars.csv.")
    costs_full = np.asarray(costs_full).reshape(-1)

    # Allow product_data to use clustering_ids instead of product_ids.
    if id_col not in product_data.columns and "clustering_ids" in product_data.columns:
        product_data = product_data.copy()
        product_data[id_col] = product_data["clustering_ids"].astype(str)

    if price_x2_index is None or beta_price_index is None:
        raise ValueError("price_x2_index and beta_price_index are required to compute manual CS.")

    cf_year, cf_costs_df, cf_cost_col = run_cf(
        results, product_data, costs_df2,
        costs_full=costs_full,
        year=year, market_col=market_col, id_col=id_col,
        parts_tariff=parts_tariff, vehicle_tariff=vehicle_tariff,
        country_tariffs=country_tariffs,
        parts_pass_through=parts_pass_through, vehicle_pass_through=vehicle_pass_through,
        iter_cfg=iter_cfg,
        firm_col=firm_col,
        ownership_mode=ownership_mode,
        owner_map=owner_map,
        pricer_map=pricer_map,
        allow_unmapped_brands=allow_unmapped_brands,
    )

    cf_ps = cf_year.set_index([market_col, id_col])[["p_cf", "s_cf"]]
    cf_cost_map = cf_costs_df.set_index(id_col)[cf_cost_col]

    aligned_pd = align_product_data_for_sim(results, product_data, id_col=id_col, market_col=market_col)
    if ownership_mode == "owner":
        if pricer_map is None:
            raise ValueError("pricer_map is required when ownership_mode='owner'.")
        aligned_pd = add_owner_ids(
            aligned_pd,
            pricer_map,
            firm_col=firm_col,
            owner_col="pricer_ids",
            allow_unmapped=allow_unmapped_brands,
        )
        if owner_map is not None:
            aligned_pd = add_owner_ids(
                aligned_pd,
                owner_map,
                firm_col=firm_col,
                owner_col="owner_ids",
                allow_unmapped=allow_unmapped_brands,
            )
    year_markets = aligned_pd.loc[aligned_pd["market_year"] == year, market_col].unique()

    prod_blocks = []

    for mid in year_markets:
        m_idx = aligned_pd[market_col].eq(mid)
        ids_m = aligned_pd.loc[m_idx, id_col].astype(str).to_numpy()

        p0 = pd.to_numeric(aligned_pd.loc[m_idx, price_col], errors="coerce").to_numpy(dtype=float)
        s0 = np.asarray(results.compute_shares(market_id=mid), dtype=float).reshape(-1)
        market_ids_full = np.asarray(results.problem.products.market_ids).reshape(-1)
        c0 = costs_full[market_ids_full == mid]

        cf_block = cf_ps.reindex(pd.MultiIndex.from_product([[mid], ids_m]))
        p_cf = pd.to_numeric(cf_block["p_cf"], errors="coerce").to_numpy(dtype=float)
        s_cf = pd.to_numeric(cf_block["s_cf"], errors="coerce").to_numpy(dtype=float)

        c_cf = pd.to_numeric(cf_cost_map.reindex(ids_m), errors="coerce").to_numpy(dtype=float)
        c_cf = np.where(np.isfinite(c_cf), c_cf, c0)

        mu0 = p0 - c0
        mu_cf = p_cf - c_cf
        pi0 = mu0 * s0
        pi_cf = mu_cf * s_cf

        base_cols = [market_col, id_col, firm_col, plant_col]
        if "owner_ids" in aligned_pd.columns:
            base_cols.append("owner_ids")
        if "pricer_ids" in aligned_pd.columns:
            base_cols.append("pricer_ids")
        block = aligned_pd.loc[m_idx, base_cols].copy()
        block[id_col] = block[id_col].astype(str)

        block["p0"] = p0
        block["p_cf"] = p_cf
        block["dp"] = p_cf - p0
        block["dp_pct"] = 100.0 * np.where(p0 != 0, block["dp"] / p0, np.nan)

        block["c0"] = c0
        block["c_cf"] = c_cf
        block["dc"] = c_cf - c0
        block["dc_pct"] = 100.0 * np.where(c0 != 0, block["dc"] / c0, np.nan)

        block["s0"] = s0
        block["s_cf"] = s_cf
        block["ds"] = s_cf - s0
        block["ds_pct"] = 100.0 * np.where(s0 != 0, block["ds"] / s0, np.nan)

        block["mu0"] = mu0
        block["mu_cf"] = mu_cf
        block["dmu"] = mu_cf - mu0

        margin0 = 100.0 * np.where(p0 != 0, mu0 / p0, np.nan)
        margin_cf = 100.0 * np.where(p_cf != 0, mu_cf / p_cf, np.nan)
        block["margin0_pct"] = margin0
        block["margin_cf_pct"] = margin_cf
        block["dmargin_pct"] = margin_cf - margin0

        block["pi0"] = pi0
        block["pi_cf"] = pi_cf
        block["dpi"] = pi_cf - pi0

        prod_blocks.append(block)

    product_table = pd.concat(prod_blocks, ignore_index=True)
    market_table = cs_manual_all_markets(
        results,
        product_table,
        price_x2_index=price_x2_index,
        beta_price_index=beta_price_index,
        gamma=gamma,
        market_col=market_col,
        id_col=id_col,
    )

    # --- firm/owner tables (share-weighted means + profit totals) ---
    def _build_group_table(group_col: str) -> pd.DataFrame:
        rows = []
        for firm, d in product_table.groupby(group_col, dropna=False):
            w0 = d["s0"].to_numpy(dtype=float)
            wcf = d["s_cf"].to_numpy(dtype=float)

            def wavg(x, w):
                den = float(np.nansum(w))
                return np.nan if den <= 0 else float(np.nansum(x.to_numpy(dtype=float) * w) / den)

            r = {
                group_col: firm,
                "share0_total": float(np.nansum(w0)),
                "share_cf_total": float(np.nansum(wcf)),
                "p0_sw": wavg(d["p0"], w0),
                "p_cf_sw": wavg(d["p_cf"], wcf),
                "c0_sw": wavg(d["c0"], w0),
                "c_cf_sw": wavg(d["c_cf"], wcf),
                "mu0_sw": wavg(d["mu0"], w0),
                "mu_cf_sw": wavg(d["mu_cf"], wcf),
                "pi0_percap_total": float(np.nansum(d["pi0"].to_numpy(dtype=float))),
                "pi_cf_percap_total": float(np.nansum(d["pi_cf"].to_numpy(dtype=float))),
            }
            r["dp_sw"] = r["p_cf_sw"] - r["p0_sw"]
            r["dc_sw"] = r["c_cf_sw"] - r["c0_sw"]
            r["dmu_sw"] = r["mu_cf_sw"] - r["mu0_sw"]
            r["dpi_percap_total"] = r["pi_cf_percap_total"] - r["pi0_percap_total"]
            rows.append(r)
        tbl = pd.DataFrame(rows)
        factor_musd = float(total_market_size) * (price_scale_usd_per_unit / 1_000_000.0)
        tbl["pi0_millions_usd"] = tbl["pi0_percap_total"] * factor_musd
        tbl["pi_cf_millions_usd"] = tbl["pi_cf_percap_total"] * factor_musd
        tbl["dpi_millions_usd"] = tbl["dpi_percap_total"] * factor_musd
        return tbl.sort_values("dpi_millions_usd", ascending=False).reset_index(drop=True)

    firm_table = _build_group_table(firm_col)
    owner_table = _build_group_table("owner_ids") if "owner_ids" in product_table.columns else None

    # --- dollars (millions) ---
    factor_musd = float(total_market_size) * (price_scale_usd_per_unit / 1_000_000.0)

    market_table["CS0_millions_usd"] = market_table["CS0"] * factor_musd
    market_table["CS_cf_millions_usd"] = market_table["CS_cf"] * factor_musd
    market_table["dCS_millions_usd"] = market_table["dCS"] * factor_musd

    overall = pd.DataFrame([{
        "year": year,
        "total_firm_surplus_change_millions_usd": float(firm_table["dpi_millions_usd"].sum()),
        "total_consumer_surplus_change_millions_usd": float(market_table["dCS_millions_usd"].sum()),
        "assumptions_total_market_size": float(total_market_size),
        "assumptions_price_scale_usd_per_unit": float(price_scale_usd_per_unit),
    }])

    return {
        "product_table": product_table,
        "firm_table": firm_table,
        "owner_table": owner_table,
        "market_surplus_table": market_table,
        "overall_surplus": overall,
        "cf_costs_df": cf_costs_df,
        "cf_cost_col": cf_cost_col,
    }


def origin_percent_metrics(
    product_tbl: pd.DataFrame,
    *,
    plant_col: str = "plant_country",
    us_value: str = "United States",
):
    df = product_tbl.copy()
    df["origin"] = np.where(df[plant_col].astype(str).eq(us_value), "US-assembled", "Foreign-assembled")

    def wavg(x, w):
        den = float(np.nansum(w))
        return np.nan if den <= 0 else float(np.nansum(x * w) / den)

    rows = []
    for o, d in df.groupby("origin", dropna=False):
        w = d["s0"].to_numpy(dtype=float)
        rows.append({
            "origin": o,
            "ΔPrice (%)": wavg(d["dp_pct"].to_numpy(dtype=float), w),
            "ΔCost (%)": wavg(d["dc_pct"].to_numpy(dtype=float), w),
            "ΔMarkup (pp)": wavg(d["dmargin_pct"].to_numpy(dtype=float), w),
            "ΔShare (pp)": 100.0 * (float(np.nansum(d["s_cf"])) - float(np.nansum(d["s0"]))),
        })

    return pd.DataFrame(rows).set_index("origin").sort_index()


def plot_origin_percent_metrics_bw(
    metrics_tbl: pd.DataFrame,
    *,
    title: str | None = None,
    show: bool = True,
):
    cats = ["ΔPrice (%)", "ΔCost (%)", "ΔMarkup (pp)", "ΔShare (pp)"]
    X = np.arange(len(cats))
    width = 0.35

    vals_us = metrics_tbl.loc["US-assembled", cats].to_numpy() if "US-assembled" in metrics_tbl.index else np.zeros(len(cats))
    vals_for = metrics_tbl.loc["Foreign-assembled", cats].to_numpy() if "Foreign-assembled" in metrics_tbl.index else np.zeros(len(cats))

    fig, ax = plt.subplots(figsize=(10, 5))

    b1 = ax.bar(X - width/2, vals_us, width, label="US-assembled",
                facecolor="white", edgecolor="black", linewidth=1.2, hatch="////")
    b2 = ax.bar(X + width/2, vals_for, width, label="Foreign-assembled",
                facecolor="0.7", edgecolor="black", linewidth=1.2)

    ax.set_xticks(X)
    ax.set_xticklabels(cats)
    ax.set_ylabel("Percent / Percentage points")
    # No figure titles
    ax.legend(frameon=False)

    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            va = "bottom" if h >= 0 else "top"
            offset = 5 if h >= 0 else -5
            ax.annotate(
                f"{h:+.2f}",
                (bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, offset),
                textcoords="offset points",
                ha="center",
                va=va,
                fontsize=9,
            )

    plt.tight_layout()
    if show:
        plt.show()
    return fig
