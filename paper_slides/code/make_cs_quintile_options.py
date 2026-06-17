#!/usr/bin/env python3
"""Build slide-ready consumer-surplus-by-quintile graph options."""

from __future__ import annotations

import re
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
GRAPH_DIR = REPO_ROOT / "paper" / "generated" / "graphs"
SLIDE_ASSET_DIR = REPO_ROOT / "paper_slides" / "output" / "assets"

SCENARIOS = [
    ("C1", "Vehicle-only\ntariff", "vehicles-only tariff (no subsidy)", "#667085"),
    ("C2", "Vehicle + parts\ntariff", "parts and vehicles tariff (no subsidy)", "#B42318"),
    ("C3", "EV subsidy", "no tariff (with subsidy)", "#0F766E"),
]
DELTA = "\N{GREEK CAPITAL LETTER DELTA}"
QUINTILES = [
    ("Q1", "Q1 lowest\nincome", f"CS {DELTA} Q1 (billion USD)"),
    ("Q2", "Q2", f"CS {DELTA} Q2 (billion USD)"),
    ("Q3", "Q3", f"CS {DELTA} Q3 (billion USD)"),
    ("Q4", "Q4", f"CS {DELTA} Q4 (billion USD)"),
    ("Q5", "Q5 highest\nincome", f"CS {DELTA} Q5 (billion USD)"),
]


def parse_cs_cell(value: str) -> tuple[float, float]:
    match = re.match(r"\s*([+-]?\d+(?:\.\d+)?)\s+\(([+-]?\d+(?:\.\d+)?)%\)\s*", str(value))
    if not match:
        raise ValueError(f"Could not parse consumer surplus cell: {value!r}")
    return float(match.group(1)), float(match.group(2))


def build_data() -> pd.DataFrame:
    summary = pd.read_csv(SAVED_OUTPUT / "summary_tbl_all.csv.gz", index_col=0)
    rows: list[dict[str, float | str]] = []
    for q_code, q_label, row_name in QUINTILES:
        for scenario, label, column, color in SCENARIOS:
            dollars, percent = parse_cs_cell(summary.loc[row_name, column])
            rows.append(
                {
                    "quintile": q_code,
                    "quintile_label": q_label,
                    "scenario": scenario,
                    "scenario_label": label.replace("\n", " "),
                    "scenario_plot_label": f"{scenario}: {label.replace(chr(10), ' ')}",
                    "color": color,
                    "delta_cs_billion_usd": dollars,
                    "delta_cs_percent": percent,
                }
            )
    return pd.DataFrame(rows)


def apply_common_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#1F2933",
            "axes.labelcolor": "#1F2933",
            "xtick.color": "#1F2933",
            "ytick.color": "#1F2933",
            "text.color": "#1F2933",
            "axes.titleweight": "bold",
        }
    )


def decorate_axis(ax, title: str, xlabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=15, pad=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.axvline(0, color="#667085", linewidth=1.0, linestyle=(0, (4, 3)))
    ax.grid(axis="x", color="#D0D5DD", linewidth=0.7, alpha=0.75)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#D0D5DD")
    ax.spines["bottom"].set_color("#D0D5DD")
    ax.tick_params(axis="both", labelsize=10)


def draw_dotplot(data: pd.DataFrame) -> None:
    apply_common_style()
    paper = "#FBFAF7"
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 7.4), facecolor=paper)
    for ax in axes:
        ax.set_facecolor(paper)

    y_base = {q: i for i, (q, _, _) in enumerate(QUINTILES)}
    offsets = {"C1": -0.15, "C2": 0.0, "C3": 0.15}
    marker_size = 115

    for ax, column, title, xlabel, xlim, fmt in [
        (
            axes[0],
            "delta_cs_percent",
            "Percent change in consumer surplus",
            "Change in consumer surplus (%)",
            (-17.8, 5.0),
            "{:+.1f}%",
        ),
        (
            axes[1],
            "delta_cs_billion_usd",
            "Dollar change in consumer surplus",
            "Change in consumer surplus (billion USD)",
            (-14.7, 5.4),
            "{:+.1f}",
        ),
    ]:
        for scenario, label, _, color in SCENARIOS:
            subset = data.loc[data["scenario"].eq(scenario)].copy()
            y = [y_base[q] + offsets[scenario] for q in subset["quintile"]]
            ax.scatter(
                subset[column],
                y,
                s=marker_size,
                color=color,
                edgecolor="white",
                linewidth=0.9,
                label=f"{scenario}: {label.replace(chr(10), ' ')}",
                zorder=3,
            )
            for value, y_pos in zip(subset[column], y):
                ha = "left" if value >= 0 else "right"
                pad = 0.22 if column == "delta_cs_percent" else 0.16
                ax.text(
                    value + (pad if value >= 0 else -pad),
                    y_pos,
                    fmt.format(value),
                    ha=ha,
                    va="center",
                    fontsize=8.7,
                    color=color,
                    fontweight="bold",
                )
        decorate_axis(ax, title, xlabel)
        ax.set_xlim(*xlim)
        ax.set_yticks(range(len(QUINTILES)))
        ax.set_yticklabels([q_label for _, q_label, _ in QUINTILES])
        ax.set_ylim(-0.55, len(QUINTILES) - 0.45)

    axes[1].set_yticklabels([])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.875),
        ncol=3,
        frameon=False,
        fontsize=10,
    )
    fig.text(
        0.055,
        0.955,
        "Option 1: consumer surplus impacts differ by income margin",
        fontsize=23,
        fontweight="bold",
        ha="left",
    )
    fig.text(
        0.055,
        0.915,
        "Percent changes show burden among buyers; dollar changes also reflect purchase volume.",
        fontsize=12,
        color="#667085",
        ha="left",
    )
    fig.subplots_adjust(left=0.12, right=0.97, top=0.78, bottom=0.12, wspace=0.18)
    save_outputs(fig, data, "cs_quintile_option1_dotplot")


def draw_grouped_bars(data: pd.DataFrame) -> None:
    apply_common_style()
    paper = "#FBFAF7"
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 7.4), facecolor=paper)
    for ax in axes:
        ax.set_facecolor(paper)

    y_base = {q: i for i, (q, _, _) in enumerate(QUINTILES)}
    offsets = {"C1": -0.22, "C2": 0.0, "C3": 0.22}
    bar_height = 0.19

    for ax, column, title, xlabel, xlim, fmt in [
        (
            axes[0],
            "delta_cs_percent",
            "Percent change in consumer surplus",
            "Change in consumer surplus (%)",
            (-17.8, 5.0),
            "{:+.1f}%",
        ),
        (
            axes[1],
            "delta_cs_billion_usd",
            "Dollar change in consumer surplus",
            "Change in consumer surplus (billion USD)",
            (-14.7, 5.4),
            "{:+.1f}",
        ),
    ]:
        for scenario, label, _, color in SCENARIOS:
            subset = data.loc[data["scenario"].eq(scenario)].copy()
            y = [y_base[q] + offsets[scenario] for q in subset["quintile"]]
            ax.barh(
                y,
                subset[column],
                height=bar_height,
                color=color,
                edgecolor="white",
                linewidth=0.7,
                label=f"{scenario}: {label.replace(chr(10), ' ')}",
            )
            for value, y_pos in zip(subset[column], y):
                ha = "left" if value >= 0 else "right"
                pad = 0.20 if column == "delta_cs_percent" else 0.15
                ax.text(
                    value + (pad if value >= 0 else -pad),
                    y_pos,
                    fmt.format(value),
                    ha=ha,
                    va="center",
                    fontsize=8.4,
                    color=color,
                    fontweight="bold",
                )
        decorate_axis(ax, title, xlabel)
        ax.set_xlim(*xlim)
        ax.set_yticks(range(len(QUINTILES)))
        ax.set_yticklabels([q_label for _, q_label, _ in QUINTILES])
        ax.set_ylim(-0.6, len(QUINTILES) - 0.4)

    axes[1].set_yticklabels([])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.875),
        ncol=3,
        frameon=False,
        fontsize=10,
    )
    fig.text(
        0.055,
        0.955,
        "Option 2: consumer surplus changes by income quintile",
        fontsize=23,
        fontweight="bold",
        ha="left",
    )
    fig.text(
        0.055,
        0.915,
        "Grouped bars compare the tariff and subsidy scenarios within each income quintile.",
        fontsize=12,
        color="#667085",
        ha="left",
    )
    fig.subplots_adjust(left=0.12, right=0.97, top=0.78, bottom=0.12, wspace=0.18)
    save_outputs(fig, data, "cs_quintile_option2_grouped_bars")


def draw_final_grouped_bars(data: pd.DataFrame) -> None:
    apply_common_style()
    paper = "#FBFAF7"
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.7), facecolor=paper)
    for ax in axes:
        ax.set_facecolor(paper)

    y_base = {q: i for i, (q, _, _) in enumerate(QUINTILES)}
    offsets = {"C1": -0.22, "C2": 0.0, "C3": 0.22}
    bar_height = 0.19

    for ax, column, title, xlabel, xlim, fmt in [
        (
            axes[0],
            "delta_cs_percent",
            "Percent change",
            "Change in consumer surplus (%)",
            (-17.8, 5.0),
            "{:+.1f}%",
        ),
        (
            axes[1],
            "delta_cs_billion_usd",
            "Dollar change",
            "Change in consumer surplus (billion USD)",
            (-14.7, 5.4),
            "{:+.1f}",
        ),
    ]:
        for scenario, label, _, color in SCENARIOS:
            subset = data.loc[data["scenario"].eq(scenario)].copy()
            y = [y_base[q] + offsets[scenario] for q in subset["quintile"]]
            ax.barh(
                y,
                subset[column],
                height=bar_height,
                color=color,
                edgecolor="white",
                linewidth=0.7,
                label=f"{scenario}: {label.replace(chr(10), ' ')}",
            )
            for value, y_pos in zip(subset[column], y):
                ha = "left" if value >= 0 else "right"
                pad = 0.20 if column == "delta_cs_percent" else 0.15
                ax.text(
                    value + (pad if value >= 0 else -pad),
                    y_pos,
                    fmt.format(value),
                    ha=ha,
                    va="center",
                    fontsize=8.3,
                    color=color,
                    fontweight="bold",
                )
        decorate_axis(ax, title, xlabel)
        ax.set_xlim(*xlim)
        ax.set_yticks(range(len(QUINTILES)))
        ax.set_yticklabels([q_label for _, q_label, _ in QUINTILES])
        ax.set_ylim(-0.6, len(QUINTILES) - 0.4)

    axes[1].set_yticklabels([])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.965),
        ncol=3,
        frameon=False,
        fontsize=10,
    )
    fig.subplots_adjust(left=0.105, right=0.985, top=0.84, bottom=0.16, wspace=0.18)
    save_outputs(fig, data, "cs_quintile_grouped_bars")


def save_outputs(fig, data: pd.DataFrame, stem: str) -> None:
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    SLIDE_ASSET_DIR.mkdir(parents=True, exist_ok=True)
    png_graph = GRAPH_DIR / f"{stem}.png"
    pdf_graph = GRAPH_DIR / f"{stem}.pdf"
    png_slide = SLIDE_ASSET_DIR / f"{stem}.png"
    csv_slide = SLIDE_ASSET_DIR / f"{stem}.csv"
    fig.savefig(png_graph, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(pdf_graph, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(png_slide, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    data.to_csv(csv_slide, index=False)
    plt.close(fig)
    print(png_graph)
    print(pdf_graph)
    print(png_slide)
    print(csv_slide)


def main() -> None:
    data = build_data()
    draw_dotplot(data)
    draw_grouped_bars(data)
    draw_final_grouped_bars(data)


if __name__ == "__main__":
    main()
