"""counterfactual_reporting.py

Helpers to run standardized counterfactual scenarios and build common tables/figures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from helpers.counterfactual_helpers import run_unified_counterfactual
from helpers.counterfactual_profit_tables import profit_changes_table, profit_changes_table_latex
from helpers.counterfactual_helpers import origin_percent_metrics, plot_origin_percent_metrics_bw
from helpers.consumer_surplus import cs_change_by_state


@dataclass(frozen=True)
class ScenarioSpec:
    label: str
    parts_tariff: float
    vehicle_tariff: float
    subsidy_zero: bool
    country_tariffs: dict[str, float] | None = None


def default_scenario_specs(
    *,
    parts_tariff: float,
    vehicle_tariff: float,
    country_tariffs: dict[str, float] | None = None,
) -> dict[str, ScenarioSpec]:
    return {
        "tariffs_parts_vehicles_with_subsidy": ScenarioSpec(
            label="parts and vehicles tariff (with subsidy)",
            parts_tariff=parts_tariff,
            vehicle_tariff=vehicle_tariff,
            subsidy_zero=False,
            country_tariffs=country_tariffs,
        ),
        "tariffs_vehicles_only_with_subsidy": ScenarioSpec(
            label="vehicles-only tariff (with subsidy)",
            parts_tariff=0.0,
            vehicle_tariff=vehicle_tariff,
            subsidy_zero=False,
            country_tariffs=country_tariffs,
        ),
        "tariffs_parts_vehicles_no_subsidy": ScenarioSpec(
            label="parts and vehicles tariff (no subsidy)",
            parts_tariff=parts_tariff,
            vehicle_tariff=vehicle_tariff,
            subsidy_zero=True,
            country_tariffs=country_tariffs,
        ),
        "tariffs_vehicles_only_no_subsidy": ScenarioSpec(
            label="vehicles-only tariff (no subsidy)",
            parts_tariff=0.0,
            vehicle_tariff=vehicle_tariff,
            subsidy_zero=True,
            country_tariffs=country_tariffs,
        ),
        "no_tariff_with_subsidy": ScenarioSpec(
            label="no tariff (with subsidy)",
            parts_tariff=0.0,
            vehicle_tariff=0.0,
            subsidy_zero=False,
            country_tariffs=None,
        ),
        "no_tariff_no_subsidy": ScenarioSpec(
            label="no tariff (no subsidy)",
            parts_tariff=0.0,
            vehicle_tariff=0.0,
            subsidy_zero=True,
            country_tariffs=None,
        ),
    }


def run_scenario_outputs(
    results,
    product_data: pd.DataFrame,
    costs_df2: pd.DataFrame,
    *,
    agent_data,
    year: int,
    price_x2_index: int,
    beta_price_index: int,
    gamma: float = 0.0,
    total_market_size: float = 132_000_000 / 6,
    price_scale_usd_per_unit: float = 100_000.0,
    specs: dict[str, ScenarioSpec],
) -> dict[str, dict[str, Any]]:
    outputs: dict[str, dict[str, Any]] = {}
    for key, spec in specs.items():
        out = run_unified_counterfactual(
            results,
            product_data,
            costs_df2,
            agent_data=agent_data,
            year=year,
            parts_tariff=spec.parts_tariff,
            vehicle_tariff=spec.vehicle_tariff,
            subsidy_zero=spec.subsidy_zero,
            country_tariffs=spec.country_tariffs,
            price_x2_index=price_x2_index,
            beta_price_index=beta_price_index,
            gamma=gamma,
            total_market_size=total_market_size,
            price_scale_usd_per_unit=price_scale_usd_per_unit,
        )
        outputs[key] = {"label": spec.label, "out": out}
    return outputs


def build_profit_change_artifacts(
    firm_table: pd.DataFrame,
    *,
    scenario_label: str,
    us_firms: list[str] | None = None,
    n: int = 5,
) -> dict[str, Any]:
    tbl = profit_changes_table(firm_table, n=n)
    latex = profit_changes_table_latex(firm_table, n=n)

    plot_df = firm_table.copy()
    plot_df["firm_lower"] = plot_df["firm_ids"].astype(str).str.lower()
    if us_firms is None:
        us_firms = [
            "ford", "chevrolet", "gmc", "buick", "cadillac", "chrysler",
            "ram", "jeep", "dodge", "tesla", "rivian", "lucid",
        ]
    us_set = {f.lower() for f in us_firms}
    plot_df["is_us"] = plot_df["firm_lower"].isin(us_set)
    plot_df = plot_df.sort_values("dpi_millions_usd", ascending=False)
    colors = ["tab:blue" if is_us else "0.7" for is_us in plot_df["is_us"]]
    base = pd.to_numeric(plot_df["pi0_millions_usd"], errors="coerce").to_numpy(dtype=float)
    dlt = pd.to_numeric(plot_df["dpi_millions_usd"], errors="coerce").to_numpy(dtype=float)
    pct = np.full(len(plot_df), np.nan, dtype=float)
    ok = np.isfinite(base) & (base != 0)
    pct[ok] = 100.0 * dlt[ok] / base[ok]

    firm_labels = plot_df["firm_ids"].astype(str).replace({"mercedesbenz": "mercedes"})
    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(firm_labels, plot_df["dpi_millions_usd"], color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    # No figure titles
    ax.set_ylabel("Δ Profit (millions USD)")
    ax.set_xlabel("Firm")
    ax.tick_params(axis="x", rotation=45, labelsize=8)

    # annotate only leftmost/rightmost bars (already sorted by dpi_millions_usd)
    n_label = 6
    if len(dlt) <= 2 * n_label:
        label_idx = set(range(len(dlt)))
    else:
        label_idx = set(range(n_label)) | set(range(len(dlt) - n_label, len(dlt)))

    for i, (bar, pct_val) in enumerate(zip(bars, pct)):
        if i not in label_idx:
            continue
        if not np.isfinite(pct_val):
            continue
        h = bar.get_height()
        va = "bottom" if h >= 0 else "top"
        offset = 5 if h >= 0 else -5
        ax.annotate(
            f"{abs(pct_val):.0f}%",
            (bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=8,
        )

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="tab:blue", label="US-headquartered"),
        Patch(facecolor="0.7", label="Non-US"),
    ]
    ax.legend(handles=legend_handles, frameon=False, loc="upper right")
    fig.tight_layout()

    return {"table": tbl, "latex": latex, "figure": fig}


def build_origin_metrics_artifacts(
    product_table: pd.DataFrame,
    *,
    scenario_label: str,
) -> dict[str, Any]:
    metrics = origin_percent_metrics(product_table)
    fig = plot_origin_percent_metrics_bw(
        metrics,
        title=None,
    )
    return {"table": metrics, "figure": fig}


def build_state_units_table(
    product_table: pd.DataFrame,
    product_data: pd.DataFrame,
    *,
    total_market_size: float,
    market_col: str = "market_ids",
    id_col: str = "product_ids",
    plant_country_col: str = "plant_country",
    plant_location_col: str = "plant_location",
) -> pd.DataFrame:
    pt = product_table.copy()
    pt = pt.drop(columns=[c for c in [plant_country_col, plant_location_col] if c in pt.columns])

    cols = [market_col, id_col, plant_country_col, plant_location_col]
    pd_map = product_data[cols].drop_duplicates([market_col, id_col]).copy()
    m = pt.merge(pd_map, on=[market_col, id_col], how="left")

    if plant_country_col not in m.columns or plant_location_col not in m.columns:
        raise KeyError(
            f"{plant_country_col}/{plant_location_col} not found after merge; available: {','.join(m.columns)}"
        )

    us_mask = m[plant_country_col].astype(str).str.strip() == "United States"
    m_us = m.loc[us_mask & m[plant_location_col].notna()].copy()

    g = (
        m_us.groupby(plant_location_col, dropna=False)
        .apply(
            lambda d: pd.Series(
                {
                    "s0": float(np.nansum(d["s0"].to_numpy(dtype=float))),
                    "s_cf": float(np.nansum(d["s_cf"].to_numpy(dtype=float))),
                    "ds": float(np.nansum((d["s_cf"] - d["s0"]).to_numpy(dtype=float))),
                }
            )
        )
        .reset_index()
    )

    g["units_base"] = total_market_size * g["s0"]
    g["units_cf"] = total_market_size * g["s_cf"]
    g["delta_units"] = total_market_size * g["ds"]
    g["pct_change"] = np.where(g["units_base"] != 0, 100.0 * g["delta_units"] / g["units_base"], np.nan)
    return g[[plant_location_col, "units_base", "units_cf", "delta_units", "pct_change"]].sort_values("delta_units")


def build_state_cs_table(
    out: dict[str, Any],
    agent_df: pd.DataFrame,
    *,
    results,
    market_id: int,
    price_x2_index: int,
    beta_price_index: int,
    income_col: str = "log_income_10k",
    division_cols: list[str] | None = None,
    state_col: str = "state",
    year_col: str = "year",
    nodes_prefix: str = "nodes",
    weight_col: str = "weights",
    gamma: float = 0.0,
) -> pd.DataFrame:
    return cs_change_by_state(
        results,
        out["product_table"],
        agent_df,
        market_id,
        price_x2_index=price_x2_index,
        beta_price_index=beta_price_index,
        income_col=income_col,
        division_cols=division_cols,
        state_col=state_col,
        year_col=year_col,
        nodes_prefix=nodes_prefix,
        weight_col=weight_col,
        gamma=gamma,
    )


def build_state_cs_map_figure(
    state_cs_table: pd.DataFrame,
    *,
    scenario_label: str,
    state_abbr: dict[str, str],
    state_centroids: dict[str, tuple[float, float]],
) -> Any:
    if state_cs_table.empty:
        return None
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None

    state_units = state_cs_table.rename(columns={
        "state": "state_name",
        "pct_change_vs_baseline": "pct_change",
    })
    state_units = state_units[["state_name", "pct_change"]].copy()

    state_key = state_units["state_name"].astype(str).str.strip().str.title()
    state_units["state_abbr"] = state_key.map(state_abbr)
    state_units = state_units.dropna(subset=["state_abbr"]).copy()

    all_states = list(state_abbr.values())
    zmax = 12.0

    fig = go.Figure()
    fig.add_trace(
        go.Choropleth(
            locations=all_states,
            locationmode="USA-states",
            z=[0] * len(all_states),
            colorscale=[[0, "#e0e0e0"], [1, "#e0e0e0"]],
            showscale=False,
            marker_line_color="white",
            marker_line_width=0.5,
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Choropleth(
            locations=state_units["state_abbr"],
            locationmode="USA-states",
            z=state_units["pct_change"],
            colorscale="RdBu",
            zmid=0,
            zmin=-zmax,
            zmax=zmax,
            colorbar=dict(
                title="% change",
                len=0.6,
                thickness=12,
                x=0.92,
                xanchor="left",
            ),
            marker_line_color="white",
            marker_line_width=0.5,
            hovertemplate="%{location}: %{z:.2f}%<extra></extra>",
        )
    )

    label_states = [s for s in state_units["state_abbr"].tolist() if s not in {"RI", "DE"}]
    label_offsets = {
        "VT": (0.0, -0.5),
        "NH": (0.4, 0.0),
        "NY": (0.4, -0.5),
        "NJ": (0.0, -0.5),
        "MA": (0.4, -0.5),
        "CT": (0.0, -0.5),
    }
    label_lats = []
    label_lons = []
    for s in label_states:
        lat, lon = state_centroids[s]
        dlat, dlon = label_offsets.get(s, (0.0, 0.0))
        label_lats.append(lat + dlat)
        label_lons.append(lon + dlon)
    label_vals = state_units.set_index("state_abbr").loc[label_states, "pct_change"].to_numpy(dtype=float)
    label_text = [f"{s}<br>{v:+.1f}%" for s, v in zip(label_states, label_vals)]
    label_colors = [
        "white" if abs(v) >= 0.6 * zmax else "black"
        for v in label_vals
    ]
    fig.add_trace(
        go.Scattergeo(
            locationmode="USA-states",
            lat=label_lats,
            lon=label_lons,
            text=label_text,
            mode="text",
            textfont=dict(size=9, color=label_colors),
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        title=None,
        geo_scope="usa",
        margin={"r": 10, "t": 30, "l": 10, "b": 10},
        width=1100,
        height=700,
    )
    fig.update_geos(
        projection_scale=1.05,
        showframe=False,
        showcountries=False,
        showcoastlines=False,
    )
    return fig


def build_state_map_figure(
    state_units: pd.DataFrame,
    *,
    scenario_label: str,
    state_abbr: dict[str, str],
    state_centroids: dict[str, tuple[float, float]],
) -> Any:
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None

    state_map = state_units.copy()
    state_key = state_map.iloc[:, 0].astype(str).str.strip().str.title()
    state_map["state_abbr"] = state_key.map(state_abbr)
    state_map = state_map.dropna(subset=["state_abbr"]).copy()
    state_map = state_map[state_map["state_abbr"] != "AZ"].copy()

    all_states = list(state_abbr.values())
    abs_vals = np.abs(state_map["pct_change"].to_numpy(dtype=float))
    abs_vals = abs_vals[np.isfinite(abs_vals)]
    if len(abs_vals) == 0:
        zmax = 1.0
    else:
        zmax = np.nanpercentile(abs_vals, 90)
        if not np.isfinite(zmax) or zmax <= 0:
            zmax = np.nanmax(abs_vals)
        zmax = max(zmax, 40.0)

    fig = go.Figure()
    fig.add_trace(
        go.Choropleth(
            locations=all_states,
            locationmode="USA-states",
            z=[0] * len(all_states),
            colorscale=[[0, "#e0e0e0"], [1, "#e0e0e0"]],
            showscale=False,
            marker_line_color="white",
            marker_line_width=0.5,
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Choropleth(
            locations=state_map["state_abbr"],
            locationmode="USA-states",
            z=state_map["pct_change"],
            colorscale="RdBu",
            zmid=0,
            zmin=-zmax,
            zmax=zmax,
            colorbar=dict(
                title="% change",
                len=0.6,
                thickness=12,
                x=0.92,
                xanchor="left",
            ),
            marker_line_color="white",
            marker_line_width=0.5,
            hovertemplate="%{location}: %{z:.2f}%<extra></extra>",
        )
    )

    label_states = [s for s in state_map["state_abbr"].tolist() if s not in {"RI", "DE"}]
    label_offsets = {
        "VT": (0.0, -0.5),
        "NH": (0.4, 0.0),
        "NY": (0.4, -0.5),
        "NJ": (0.0, -0.5),
        "MA": (0.4, -0.5),
        "CT": (0.0, -0.5),
    }
    label_lats = []
    label_lons = []
    for s in label_states:
        lat, lon = state_centroids[s]
        dlat, dlon = label_offsets.get(s, (0.0, 0.0))
        label_lats.append(lat + dlat)
        label_lons.append(lon + dlon)
    units_millions = None
    if "units_cf" in state_map.columns:
        units_millions = state_map["units_cf"].to_numpy(dtype=float) / 1_000_000.0

    label_vals = state_map.set_index("state_abbr").loc[label_states, "pct_change"].to_numpy(dtype=float)
    label_text = []
    for s, v, u in zip(
        label_states,
        label_vals,
        units_millions if units_millions is not None else [None] * len(label_states),
    ):
        if u is None or not np.isfinite(u):
            label_text.append(f"{s}<br>{v:+.1f}%")
        else:
            label_text.append(f"{s}<br>{v:+.1f}%<br>{u:.2f}M")
    label_colors = [
        "white" if abs(v) >= 0.6 * zmax else "black"
        for v in label_vals
    ]
    fig.add_trace(
        go.Scattergeo(
            locationmode="USA-states",
            lat=label_lats,
            lon=label_lons,
            text=label_text,
            mode="text",
            textfont=dict(size=9, color=label_colors),
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        title=None,
        geo_scope="usa",
        margin={"r": 10, "t": 30, "l": 10, "b": 10},
        width=1100,
        height=700,
    )
    fig.update_geos(
        projection_scale=1.05,
        showframe=False,
        showcountries=False,
        showcoastlines=False,
    )
    return fig


def build_scenario_report(
    out: dict[str, Any],
    *,
    scenario_label: str,
    product_data: pd.DataFrame,
    total_market_size: float,
    state_cs_agent_df: pd.DataFrame | None = None,
    results=None,
    market_id: int | None = None,
    price_x2_index: int | None = None,
    beta_price_index: int | None = None,
    income_col: str = "log_income_10k",
    division_cols: list[str] | None = None,
    state_col: str = "state",
    year_col: str = "year",
    nodes_prefix: str = "nodes",
    weight_col: str = "weights",
    gamma: float = 0.0,
    state_abbr: dict[str, str] | None = None,
    state_centroids: dict[str, tuple[float, float]] | None = None,
    us_firms: list[str] | None = None,
    n_top: int = 5,
) -> dict[str, Any]:
    product_table = out["product_table"]
    firm_table = out["firm_table"]

    origin = build_origin_metrics_artifacts(product_table, scenario_label=scenario_label)
    profit = build_profit_change_artifacts(
        firm_table,
        scenario_label=scenario_label,
        us_firms=us_firms,
        n=n_top,
    )
    state_units = build_state_units_table(
        product_table,
        product_data,
        total_market_size=total_market_size,
    )
    state_fig = None
    if state_abbr is not None and state_centroids is not None:
        state_fig = build_state_map_figure(
            state_units,
            scenario_label=scenario_label,
            state_abbr=state_abbr,
            state_centroids=state_centroids,
        )

    state_cs_table = None
    state_cs_fig = None
    if state_cs_agent_df is not None:
        if results is None or market_id is None or price_x2_index is None or beta_price_index is None:
            raise ValueError("results, market_id, price_x2_index, and beta_price_index are required for state CS.")
        state_cs_table = build_state_cs_table(
            out,
            state_cs_agent_df,
            results=results,
            market_id=market_id,
            price_x2_index=price_x2_index,
            beta_price_index=beta_price_index,
            income_col=income_col,
            division_cols=division_cols,
            state_col=state_col,
            year_col=year_col,
            nodes_prefix=nodes_prefix,
            weight_col=weight_col,
            gamma=gamma,
        )
        if state_abbr is not None and state_centroids is not None:
            state_cs_fig = build_state_cs_map_figure(
                state_cs_table,
                scenario_label=scenario_label,
                state_abbr=state_abbr,
                state_centroids=state_centroids,
            )

    return {
        "origin_table": origin["table"],
        "origin_figure": origin["figure"],
        "profit_table": profit["table"],
        "profit_latex": profit["latex"],
        "profit_figure": profit["figure"],
        "state_units_table": state_units,
        "state_map_figure": state_fig,
        "state_cs_table": state_cs_table,
        "state_cs_map_figure": state_cs_fig,
    }
