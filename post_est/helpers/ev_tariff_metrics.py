
"""
ev_tariff_metrics.py

Compute (i) EV share and (ii) tariff revenue for counterfactual scenarios produced by
run_cf_and_summarize(...).

Assumptions (matching your pipeline):
- product_table has: ['market_ids','product_ids','p0','p_cf','c0','c_cf','s0','s_cf', ...]
- product_data has:  ['market_ids','product_ids','ev','plant_country', ...]
- costs_df2 has:     ['product_ids','pcUSCA_pct', ...]   (pcUSCA share, used for parts-tariff revenue base)
- prices/costs are in "USD per 100k" model units, so multiply by price_scale_usd_per_unit to get USD.
- shares are per-capita market shares; total units = total_market_size * sum(shares).

Tariff revenue accounting (simple, transparent proxies):
- Vehicle import tariff revenue (foreign-assembled):  vehicle_tariff * (pre-tariff value) * quantity
  We proxy pre-tariff value with c0 (baseline marginal cost from results).
- Imported-parts tariff revenue (US-assembled): parts_tariff * (imported parts value) * quantity
  We proxy imported parts value with (1 - pcUSCA_pct) * c0.

If you later get a better measure of import value (e.g., observed import price / FOB),
swap out the "base_value" terms below.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


_US_TOKENS = {"united states", "united states of america", "usa", "us", "u.s."}


def _is_us_country(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    return s.isin(_US_TOKENS)


def _get_market_block(product_table: pd.DataFrame, market_id) -> pd.DataFrame:
    m = product_table[product_table["market_ids"].astype(str) == str(market_id)].copy()
    if m.empty:
        raise ValueError(f"No rows in product_table for market_id={market_id}.")
    return m


def _attach_ev_and_pcshare(
    m: pd.DataFrame,
    product_data: pd.DataFrame,
    costs_df2: pd.DataFrame,
) -> pd.DataFrame:
    # ev + (potentially) plant_country / plant_location from product_data
    # Some downstream code expects a "plant_country" column; if it's not present in the
    # counterfactual product_table, we can backfill it from product_data.
    cols = ["market_ids", "product_ids", "ev"]
    for c in ("plant_country", "plant_location"):
        if c in product_data.columns:
            cols.append(c)

    pd_map = product_data[cols].drop_duplicates(["market_ids", "product_ids"]).copy()
    out = m.merge(pd_map, on=["market_ids", "product_ids"], how="left", suffixes=("", "_pd"))

    def _coalesce(base: pd.Series, other: pd.Series | None) -> pd.Series:
        return base if other is None else base.combine_first(other)

    # Ensure we end up with a single "plant_country" column (fall back to plant_location if needed)
    if "plant_country" not in out.columns:
        out["plant_country"] = np.nan
    out["plant_country"] = _coalesce(out["plant_country"], out.get("plant_country_pd"))
    out["plant_country"] = _coalesce(out["plant_country"], out.get("plant_location"))
    out["plant_country"] = _coalesce(out["plant_country"], out.get("plant_location_pd"))

    # Drop helper columns if present
    for c in ("plant_country_pd", "plant_location_pd"):
        if c in out.columns:
            out = out.drop(columns=c)

    # pcUSCA_pct from costs_df2 (by product_ids only)
    if "pcUSCA_pct" in costs_df2.columns:
        pc_map = costs_df2[["product_ids", "pcUSCA_pct"]].drop_duplicates("product_ids")
        out = out.merge(pc_map, on="product_ids", how="left")
    else:
        out["pcUSCA_pct"] = np.nan

    # clean types
    out["ev"] = pd.to_numeric(out["ev"], errors="coerce").fillna(0.0)
    out["pcUSCA_pct"] = pd.to_numeric(out["pcUSCA_pct"], errors="coerce")
    return out


def ev_share_from_shares(df: pd.DataFrame, share_col: str) -> float:
    """Share of EVs among purchases: sum(s * ev) / sum(s)."""
    s = df[share_col].to_numpy(dtype=float)
    ev = df["ev"].to_numpy(dtype=float)
    denom = np.nansum(s)
    if denom <= 0:
        return np.nan
    return float(np.nansum(s * ev) / denom)


def tariff_revenue_percap(
    df: pd.DataFrame,
    *,
    parts_tariff: float,
    vehicle_tariff: float,
    country_tariffs: dict[str, float] | None = None,
    cost_base_col: str = "c0",
) -> dict[str, float]:
    """
    Return per-capita tariff revenue components in *model units* (USD/100k):
      - parts_rev_percap
      - vehicle_rev_percap
      - total_rev_percap
    """
    c0 = df[cost_base_col].to_numpy(dtype=float)
    s_cf = df["s_cf"].to_numpy(dtype=float)

    # assembly masks
    us_assembled = _is_us_country(df["plant_country"]).to_numpy()
    foreign_assembled = ~us_assembled

    # parts: US-assembled only, base = (1 - pcUSCA_pct) * c0
    imported_share = (1.0 - df["pcUSCA_pct"].to_numpy(dtype=float))
    imported_share = np.clip(imported_share, 0.0, 1.0)
    imported_share = np.nan_to_num(imported_share, nan=0.0)

    parts_base_value = imported_share * c0
    parts_rev = parts_tariff * np.nansum(parts_base_value[us_assembled] * s_cf[us_assembled])

    # vehicles: foreign-assembled only, base = c0
    if country_tariffs:
        # map per-country rates, fallback to vehicle_tariff
        country = df["plant_country"].astype(str).str.strip()
        rates = country.map(country_tariffs).fillna(vehicle_tariff).to_numpy(dtype=float)
        rates = np.where(us_assembled, 0.0, rates)
        vehicle_rev = float(np.nansum(rates * c0 * s_cf))
    else:
        vehicle_rev = vehicle_tariff * np.nansum(c0[foreign_assembled] * s_cf[foreign_assembled])

    return {
        "parts_rev_percap": float(parts_rev),
        "vehicle_rev_percap": float(vehicle_rev),
        "total_rev_percap": float(parts_rev + vehicle_rev),
    }


def scenario_ev_and_revenue(
    out: dict,
    *,
    scenario_name: str,
    product_data: pd.DataFrame,
    costs_df2: pd.DataFrame,
    market_id,
    parts_tariff: float,
    vehicle_tariff: float,
    country_tariffs: dict[str, float] | None = None,
    total_market_size: float,
    price_scale_usd_per_unit: float = 100_000.0,
) -> pd.Series:
    """
    Build one-row summary for a scenario (baseline + CF outcomes).
    """
    pt = out["product_table"]
    m = _get_market_block(pt, market_id)
    m = _attach_ev_and_pcshare(m, product_data, costs_df2)

    # EV shares
    ev0 = ev_share_from_shares(m, "s0")
    evcf = ev_share_from_shares(m, "s_cf")
    dev_pp = 100.0 * (evcf - ev0)  # percentage points

    # Units
    units0 = float(total_market_size) * float(np.nansum(m["s0"].to_numpy(dtype=float)))
    unitscf = float(total_market_size) * float(np.nansum(m["s_cf"].to_numpy(dtype=float)))

    # Tariff revenue (per-capita model units -> USD -> millions USD)
    rev = tariff_revenue_percap(
        m,
        parts_tariff=parts_tariff,
        vehicle_tariff=vehicle_tariff,
        country_tariffs=country_tariffs,
        cost_base_col="c0",
    )
    factor_musd = float(total_market_size) * (float(price_scale_usd_per_unit) / 1_000_000.0)

    parts_rev_m = rev["parts_rev_percap"] * factor_musd
    veh_rev_m = rev["vehicle_rev_percap"] * factor_musd
    tot_rev_m = rev["total_rev_percap"] * factor_musd

    return pd.Series({
        "Scenario": scenario_name,
        "EV share (baseline)": ev0,
        "EV share (CF)": evcf,
        "Δ EV share (pp)": dev_pp,
        "Units (baseline)": units0,
        "Units (CF)": unitscf,
        "Tariff revenue – vehicles (million USD)": veh_rev_m,
        "Tariff revenue – parts (million USD)": parts_rev_m,
        "Tariff revenue – total (million USD)": tot_rev_m,
        "parts_tariff": parts_tariff,
        "vehicle_tariff": vehicle_tariff,
        "market_id": str(market_id),
    })


def build_ev_and_tariff_table(
    scenarios: dict,
    *,
    product_data: pd.DataFrame,
    costs_df2: pd.DataFrame,
    market_id,
    total_market_size: float,
    price_scale_usd_per_unit: float = 100_000.0,
) -> pd.DataFrame:
    """
    scenarios: dict like
      {
        "25% Vehicles Only": {"out": out_veh, "vehicle_tariff": 0.25, "parts_tariff": 0.0},
        "25% Vehicles & Parts": {"out": out_both, "vehicle_tariff": 0.25, "parts_tariff": 0.25},
      }
    """
    rows = []
    for name, spec in scenarios.items():
        rows.append(
            scenario_ev_and_revenue(
                spec["out"],
                scenario_name=name,
                product_data=product_data,
                costs_df2=costs_df2,
                market_id=market_id,
                parts_tariff=float(spec.get("parts_tariff", 0.0)),
                vehicle_tariff=float(spec.get("vehicle_tariff", 0.0)),
                country_tariffs=spec.get("country_tariffs"),
                total_market_size=total_market_size,
                price_scale_usd_per_unit=price_scale_usd_per_unit,
            )
        )

    df = pd.DataFrame(rows)
    # nice formatting order
    cols = [
        "Scenario",
        "EV share (baseline)", "EV share (CF)", "Δ EV share (pp)",
        "Units (baseline)", "Units (CF)",
        "Tariff revenue – vehicles (million USD)",
        "Tariff revenue – parts (million USD)",
        "Tariff revenue – total (million USD)",
        "parts_tariff", "vehicle_tariff", "market_id",
    ]
    return df[cols]
