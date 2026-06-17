#!/usr/bin/env python3
"""Build a seminar figure with EV sales changes by firm and scenario."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SAVED_OUTPUT = (
    REPO_ROOT
    / "post_est"
    / "saved_outputs"
    / "blp_results_20260202_060248_0111110110000011010_45w_dbl_20260504_095746_984656_rebased_b0"
)
PRODUCT_DATA = REPO_ROOT / "post_est" / "data" / "raw" / "blp_with_45W_subsidies_scale1p0.csv"
GRAPH_DIR = REPO_ROOT / "paper" / "generated" / "graphs"
SLIDE_ASSET_DIR = REPO_ROOT / "paper_slides" / "output" / "assets"

SCENARIOS = [
    ("B0", "B0: no tariff,\nno subsidy", "no_tariff__no_subsidy"),
    ("C1", "C1: vehicle-only\ntariff", "vehicles_only_tariff__no_subsidy"),
    ("C2", "C2: vehicle + parts\ntariff", "parts_and_vehicles_tariff__no_subsidy"),
    ("C3", "C3: EV subsidy", "no_tariff__with_subsidy"),
    ("C4", "C4: vehicle-only\ntariff + subsidy", "vehicles_only_tariff__with_subsidy"),
    ("C5", "C5: vehicle + parts\ntariff + subsidy", "parts_and_vehicles_tariff__with_subsidy"),
]

FIRM_LABELS = {
    "tesla": "Tesla",
    "hyundai": "Hyundai/Kia",
    "generalmotors": "General Motors",
    "bmw": "BMW",
    "volkswagen": "Volkswagen",
    "ford": "Ford",
    "honda": "Honda",
    "rivian": "Rivian",
    "mercedesbenzgroup": "Mercedes-Benz",
    "toyota": "Toyota",
}


def load_ev_map() -> pd.DataFrame:
    product_data = pd.read_csv(PRODUCT_DATA)
    ev_map = product_data.loc[
        product_data["market_ids"].eq(2024),
        ["market_ids", "product_ids", "engine_type"],
    ].drop_duplicates(["market_ids", "product_ids"])
    ev_map["ev"] = ev_map["engine_type"].astype(str).str.lower().eq("electric").astype(int)
    return ev_map[["market_ids", "product_ids", "ev"]]


def build_sales_by_firm() -> tuple[pd.DataFrame, list[str]]:
    metadata = json.loads((SAVED_OUTPUT / "metadata.json").read_text())
    total_market_size = float(metadata["total_market_size"])
    ev_map = load_ev_map()

    scenario_sales: dict[str, pd.Series] = {}
    for code, _, slug in SCENARIOS:
        product_table = pd.read_csv(SAVED_OUTPUT / f"{slug}__product_table.csv.gz")
        merged = product_table.merge(ev_map, on=["market_ids", "product_ids"], how="left")
        merged["ev"] = pd.to_numeric(merged["ev"], errors="coerce").fillna(0.0)
        ev_products = merged.loc[merged["ev"].eq(1)].copy()
        scenario_sales[code] = (
            ev_products.groupby("owner_ids", dropna=False)["s_cf"].sum() * total_market_size / 1_000.0
        )

    baseline = scenario_sales["B0"].sort_values(ascending=False)
    major_firms = list(baseline.head(10).index)

    rows: list[dict[str, float | str]] = []
    for firm in major_firms:
        base_sales = float(scenario_sales["B0"].get(firm, 0.0))
        for code, label, _ in SCENARIOS:
            sales = float(scenario_sales[code].get(firm, 0.0))
            rows.append(
                {
                    "scenario": code,
                    "scenario_label": label.replace("\n", " "),
                    "owner_id": firm,
                    "firm": FIRM_LABELS.get(str(firm), str(firm).title()),
                    "ev_sales_thousands": sales,
                    "ev_sales_change_thousands_vs_b0": sales - base_sales,
                    "baseline_ev_sales_thousands": base_sales,
                }
            )

    return pd.DataFrame(rows), major_firms


def draw_figure(data: pd.DataFrame, major_firms: list[str], output_stem: str) -> None:
    ink = "#1F2933"
    muted = "#667085"
    teal = "#0F766E"
    red = "#B42318"
    gray = "#98A2B3"
    paper = "#FBFAF7"
    grid = "#D0D5DD"

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": ink,
            "axes.labelcolor": ink,
            "xtick.color": ink,
            "ytick.color": ink,
            "text.color": ink,
            "axes.titleweight": "bold",
        }
    )

    ordered_firms = list(reversed(major_firms))
    firm_labels = [FIRM_LABELS.get(str(f), str(f).title()) for f in ordered_firms]

    fig, axes = plt.subplots(2, 3, figsize=(14.2, 8.0), sharey=True)
    axes_flat = axes.ravel()
    fig.patch.set_facecolor(paper)

    max_abs_change = float(
        data.loc[~data["scenario"].eq("B0"), "ev_sales_change_thousands_vs_b0"].abs().max()
    )
    x_limit = max_abs_change * 1.18
    baseline_limit = float(data.loc[data["scenario"].eq("B0"), "ev_sales_thousands"].max()) * 1.22

    for ax, (code, label, _) in zip(axes_flat, SCENARIOS):
        panel = (
            data.loc[data["scenario"].eq(code)]
            .set_index("owner_id")
            .reindex(ordered_firms)
            .reset_index()
        )
        ax.set_facecolor(paper)
        y = range(len(panel))

        if code == "B0":
            values = panel["ev_sales_thousands"]
            bars = ax.barh(y, values, color=gray, edgecolor="white", linewidth=0.8)
            ax.set_xlim(0, baseline_limit)
            ax.set_xlabel("Thousand EVs")
            for bar, value in zip(bars, values):
                ax.text(
                    value + baseline_limit * 0.018,
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:.0f}",
                    va="center",
                    ha="left",
                    fontsize=8.5,
                    fontweight="bold",
                )
        else:
            values = panel["ev_sales_change_thousands_vs_b0"]
            colors = [teal if v >= 0 else red for v in values]
            bars = ax.barh(y, values, color=colors, edgecolor="white", linewidth=0.8)
            ax.axvline(0, color=muted, linewidth=1.0)
            ax.set_xlim(-x_limit, x_limit)
            ax.set_xlabel("Change from B0, thousand EVs")
            for bar, value in zip(bars, values):
                offset = x_limit * 0.025
                ha = "left" if value >= 0 else "right"
                x_text = value + offset if value >= 0 else value - offset
                ax.text(
                    x_text,
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:+.0f}",
                    va="center",
                    ha=ha,
                    fontsize=8.5,
                    fontweight="bold",
                )

        ax.set_title(label, fontsize=11, loc="left")
        ax.set_yticks(list(y))
        ax.set_yticklabels(firm_labels, fontsize=9)
        ax.grid(axis="x", color=grid, linewidth=0.7, alpha=0.75)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(grid)
        ax.spines["bottom"].set_color(grid)

    fig.suptitle(
        "Change in EV sales by major EV firm",
        x=0.055,
        y=0.985,
        ha="left",
        fontsize=17,
        fontweight="bold",
    )
    fig.text(
        0.055,
        0.94,
        "B0 panel shows baseline EV sales. Other panels show changes relative to B0, in thousand vehicles.",
        ha="left",
        fontsize=10,
        color=muted,
    )
    fig.subplots_adjust(left=0.14, right=0.985, top=0.885, bottom=0.085, wspace=0.18, hspace=0.34)

    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    SLIDE_ASSET_DIR.mkdir(parents=True, exist_ok=True)

    png_graph = GRAPH_DIR / f"{output_stem}.png"
    pdf_graph = GRAPH_DIR / f"{output_stem}.pdf"
    png_slide = SLIDE_ASSET_DIR / f"{output_stem}.png"
    csv_slide = SLIDE_ASSET_DIR / f"{output_stem}.csv"

    fig.savefig(png_graph, dpi=220, bbox_inches="tight", facecolor=paper)
    fig.savefig(pdf_graph, bbox_inches="tight", facecolor=paper)
    fig.savefig(png_slide, dpi=220, bbox_inches="tight", facecolor=paper)
    data.to_csv(csv_slide, index=False)
    plt.close(fig)

    print(png_graph)
    print(pdf_graph)
    print(png_slide)
    print(csv_slide)


def draw_c1_c3_slide_figure(data: pd.DataFrame, major_firms: list[str], output_stem: str) -> None:
    ink = "#1F2933"
    muted = "#667085"
    teal = "#0F766E"
    red = "#B42318"
    paper = "#FBFAF7"
    grid = "#D0D5DD"

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": ink,
            "axes.labelcolor": ink,
            "xtick.color": ink,
            "ytick.color": ink,
            "text.color": ink,
            "axes.titleweight": "bold",
        }
    )

    ordered_firms = list(reversed(major_firms))
    firm_labels = [FIRM_LABELS.get(str(f), str(f).title()) for f in ordered_firms]
    scenarios = [s for s in SCENARIOS if s[0] in {"C1", "C2", "C3"}]

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 6.5), sharey=True)
    fig.patch.set_facecolor(paper)

    max_abs_change = float(
        data.loc[data["scenario"].isin(["C1", "C2", "C3"]), "ev_sales_change_thousands_vs_b0"]
        .abs()
        .max()
    )
    x_limit = max_abs_change * 1.22

    for ax, (code, label, _) in zip(axes, scenarios):
        panel = (
            data.loc[data["scenario"].eq(code)]
            .set_index("owner_id")
            .reindex(ordered_firms)
            .reset_index()
        )
        values = panel["ev_sales_change_thousands_vs_b0"]
        colors = [teal if v >= 0 else red for v in values]
        y = range(len(panel))

        ax.set_facecolor(paper)
        bars = ax.barh(y, values, color=colors, edgecolor="white", linewidth=0.8)
        ax.axvline(0, color=muted, linewidth=1.0)
        ax.set_xlim(-x_limit, x_limit)
        ax.set_title(label, fontsize=15, loc="left")
        ax.set_xlabel("Change from B0, thousand EVs", fontsize=11)
        ax.set_yticks(list(y))
        ax.set_yticklabels(firm_labels, fontsize=12)
        ax.grid(axis="x", color=grid, linewidth=0.7, alpha=0.75)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(grid)
        ax.spines["bottom"].set_color(grid)

        for bar, value in zip(bars, values):
            offset = x_limit * 0.025
            ha = "left" if value >= 0 else "right"
            x_text = value + offset if value >= 0 else value - offset
            ax.text(
                x_text,
                bar.get_y() + bar.get_height() / 2,
                f"{value:+.0f}",
                va="center",
                ha=ha,
                fontsize=11,
                fontweight="bold",
            )

    fig.subplots_adjust(left=0.145, right=0.99, top=0.90, bottom=0.14, wspace=0.15)

    png_graph = GRAPH_DIR / f"{output_stem}.png"
    pdf_graph = GRAPH_DIR / f"{output_stem}.pdf"
    png_slide = SLIDE_ASSET_DIR / f"{output_stem}.png"
    csv_slide = SLIDE_ASSET_DIR / f"{output_stem}.csv"

    fig.savefig(png_graph, dpi=220, bbox_inches="tight", facecolor=paper)
    fig.savefig(pdf_graph, bbox_inches="tight", facecolor=paper)
    fig.savefig(png_slide, dpi=220, bbox_inches="tight", facecolor=paper)
    data.loc[data["scenario"].isin(["C1", "C2", "C3"])].to_csv(csv_slide, index=False)
    plt.close(fig)

    print(png_graph)
    print(pdf_graph)
    print(png_slide)
    print(csv_slide)


def main() -> None:
    data, major_firms = build_sales_by_firm()
    draw_figure(data, major_firms, "ev_firm_sales_change_by_scenario")
    draw_c1_c3_slide_figure(data, major_firms, "ev_firm_sales_change_c1_c3")


if __name__ == "__main__":
    main()
