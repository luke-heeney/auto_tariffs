#!/usr/bin/env python3
"""Build a seminar figure with EV share and EV unit sales by scenario."""

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
    ("B0", "No tariff\nno subsidy", "no_tariff__no_subsidy"),
    ("C1", "Vehicle-only\ntariff", "vehicles_only_tariff__no_subsidy"),
    ("C2", "Vehicle + parts\ntariff", "parts_and_vehicles_tariff__no_subsidy"),
    ("C3", "EV subsidy", "no_tariff__with_subsidy"),
    ("C4", "Vehicle-only\ntariff + subsidy", "vehicles_only_tariff__with_subsidy"),
    ("C5", "Vehicle + parts\ntariff + subsidy", "parts_and_vehicles_tariff__with_subsidy"),
]


def load_ev_map() -> pd.DataFrame:
    product_data = pd.read_csv(PRODUCT_DATA)
    ev_map = product_data.loc[
        product_data["market_ids"].eq(2024),
        ["market_ids", "product_ids", "engine_type"],
    ].drop_duplicates(["market_ids", "product_ids"])
    ev_map["ev"] = ev_map["engine_type"].astype(str).str.lower().eq("electric").astype(int)
    return ev_map[["market_ids", "product_ids", "ev"]]


def build_plot_data() -> pd.DataFrame:
    metadata = json.loads((SAVED_OUTPUT / "metadata.json").read_text())
    total_market_size = float(metadata["total_market_size"])
    ev_map = load_ev_map()

    rows: list[dict[str, float | str]] = []
    for code, label, slug in SCENARIOS:
        product_table = pd.read_csv(SAVED_OUTPUT / f"{slug}__product_table.csv.gz")
        merged = product_table.merge(ev_map, on=["market_ids", "product_ids"], how="left")
        merged["ev"] = pd.to_numeric(merged["ev"], errors="coerce").fillna(0.0)

        total_sales = float(merged["s_cf"].sum() * total_market_size)
        ev_sales = float((merged["s_cf"] * merged["ev"]).sum() * total_market_size)
        rows.append(
            {
                "scenario": code,
                "label": label,
                "total_vehicle_sales": total_sales,
                "ev_sales": ev_sales,
                "ev_sales_thousands": ev_sales / 1_000.0,
                "ev_share_pct": 100.0 * ev_sales / total_sales,
            }
        )

    data = pd.DataFrame(rows)
    base_share = float(data.loc[data["scenario"].eq("B0"), "ev_share_pct"].iloc[0])
    base_ev_sales = float(data.loc[data["scenario"].eq("B0"), "ev_sales"].iloc[0])
    data["ev_share_change_pp_vs_b0"] = data["ev_share_pct"] - base_share
    data["ev_sales_change_thousands_vs_b0"] = (data["ev_sales"] - base_ev_sales) / 1_000.0
    return data


def draw_figure(data: pd.DataFrame, output_stem: str) -> None:
    ink = "#1F2933"
    muted = "#667085"
    teal = "#0F766E"
    teal_light = "#8ECBC1"
    red = "#B42318"
    gray = "#98A2B3"
    paper = "#FBFAF7"
    grid = "#D0D5DD"

    colors = [gray, gray, red, teal, teal_light, teal]
    x = range(len(data))

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

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10.5, 6.2),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0], "hspace": 0.18},
    )
    fig.patch.set_facecolor(paper)

    share_base = float(data.loc[data["scenario"].eq("B0"), "ev_share_pct"].iloc[0])
    sales_base = float(data.loc[data["scenario"].eq("B0"), "ev_sales_thousands"].iloc[0])

    panels = [
        (axes[0], "EV share of vehicles sold", "Percent", "ev_share_pct", share_base, "{:.1f}%"),
        (axes[1], "Total EV sales", "Thousand vehicles", "ev_sales_thousands", sales_base, "{:.0f}k"),
    ]

    for ax, title, ylabel, column, baseline, fmt in panels:
        ax.set_facecolor(paper)
        bars = ax.bar(x, data[column], width=0.68, color=colors, edgecolor="white", linewidth=1.0)
        ax.axhline(baseline, color=muted, linewidth=1.1, linestyle=(0, (4, 3)))
        ax.grid(axis="y", color=grid, linewidth=0.7, alpha=0.75)
        ax.set_axisbelow(True)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", pad=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(grid)
        ax.spines["bottom"].set_color(grid)

        top = max(float(data[column].max()), baseline)
        ax.set_ylim(0, top * 1.28)
        for bar, value in zip(bars, data[column]):
            if column == "ev_sales_thousands":
                label = fmt.format(value)
            else:
                label = fmt.format(value)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + top * 0.035,
                label,
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(data["scenario"] + "\n" + data["label"], fontsize=9)

    fig.suptitle(
        "EV outcomes by counterfactual scenario",
        x=0.08,
        y=0.985,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.08,
        0.935,
        "EV share is conditional on vehicle purchase; sales are counterfactual unit sales in the 2024 market.",
        ha="left",
        fontsize=10,
        color=muted,
    )

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


def draw_compact_figure(data: pd.DataFrame, output_stem: str) -> None:
    ink = "#1F2933"
    muted = "#667085"
    teal = "#0F766E"
    teal_light = "#8ECBC1"
    red = "#B42318"
    gray = "#98A2B3"
    paper = "#FBFAF7"
    grid = "#D0D5DD"

    colors = [gray, gray, red, teal, teal_light, teal]
    x = range(len(data))

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

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11.0, 5.2),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0], "hspace": 0.23},
    )
    fig.patch.set_facecolor(paper)

    share_base = float(data.loc[data["scenario"].eq("B0"), "ev_share_pct"].iloc[0])
    sales_base = float(data.loc[data["scenario"].eq("B0"), "ev_sales_thousands"].iloc[0])

    panels = [
        (axes[0], "EV share of vehicles sold", "Percent", "ev_share_pct", share_base, "{:.1f}%"),
        (axes[1], "Total EV sales", "Thousand vehicles", "ev_sales_thousands", sales_base, "{:.0f}k"),
    ]

    for ax, title, ylabel, column, baseline, fmt in panels:
        ax.set_facecolor(paper)
        bars = ax.bar(x, data[column], width=0.68, color=colors, edgecolor="white", linewidth=1.0)
        ax.axhline(baseline, color=muted, linewidth=1.1, linestyle=(0, (4, 3)))
        ax.grid(axis="y", color=grid, linewidth=0.7, alpha=0.75)
        ax.set_axisbelow(True)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, loc="left", pad=6, fontsize=14)
        ax.tick_params(axis="y", labelsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(grid)
        ax.spines["bottom"].set_color(grid)

        top = max(float(data[column].max()), baseline)
        ax.set_ylim(0, top * 1.22)
        for bar, value in zip(bars, data[column]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + top * 0.032,
                fmt.format(value),
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )

    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(data["scenario"] + "\n" + data["label"], fontsize=9.5)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.965, bottom=0.18)

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


def main() -> None:
    data = build_plot_data()
    draw_figure(data, "ev_share_sales_by_scenario")
    draw_compact_figure(data, "ev_share_sales_by_scenario_compact")


if __name__ == "__main__":
    main()
