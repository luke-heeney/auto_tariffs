"""counterfactual_profit_tables.py

Reporting helpers for counterfactual firm profit changes.

Assumes firm_table has columns (from run_cf_and_summarize):
  - firm_ids
  - pi0_millions_usd
  - pi_cf_millions_usd
  - dpi_millions_usd

All profits are in millions of USD.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def top_bottom_profit_changes(
    firm_table: pd.DataFrame,
    *,
    n: int = 5,
    firm_col: str = "firm_ids",
    base_profit_col: str = "pi0_millions_usd",
    delta_profit_col: str = "dpi_millions_usd",
) -> dict[str, pd.DataFrame]:
    """Return panels of the n biggest increases and decreases in profit.

    %Δ is defined as 100*(ΔProfit / baseline Profit).
    """
    df = firm_table[[firm_col, base_profit_col, delta_profit_col]].copy()

    # % change relative to baseline; avoid divide-by-zero
    base = df[base_profit_col].to_numpy(dtype=float)
    dlt = df[delta_profit_col].to_numpy(dtype=float)
    pct = np.full(len(df), np.nan, dtype=float)
    ok = np.isfinite(base) & (base != 0)
    pct[ok] = 100.0 * dlt[ok] / base[ok]
    df["pct_change"] = pct

    inc = df.sort_values(delta_profit_col, ascending=False, kind="mergesort").head(n).copy()
    dec = df.sort_values(delta_profit_col, ascending=True,  kind="mergesort").head(n).copy()

    inc = inc.rename(columns={
        firm_col: "Firm",
        delta_profit_col: "Δ Profit (millions USD)",
        "pct_change": "%Δ",
    })[["Firm", "Δ Profit (millions USD)", "%Δ"]]

    dec = dec.rename(columns={
        firm_col: "Firm",
        delta_profit_col: "Δ Profit (millions USD)",
        "pct_change": "%Δ",
    })[["Firm", "Δ Profit (millions USD)", "%Δ"]]

    return {"panel_a_increases": inc, "panel_b_decreases": dec}


def profit_changes_table(
    firm_table: pd.DataFrame,
    *,
    n: int = 5,
    firm_col: str = "firm_ids",
    base_profit_col: str = "pi0_millions_usd",
    delta_profit_col: str = "dpi_millions_usd",
    digits_profit: int = 1,
    digits_pct: int = 1,
) -> pd.DataFrame:
    """One tidy table with a Panel column, formatted like the paper figure."""
    panels = top_bottom_profit_changes(
        firm_table,
        n=n,
        firm_col=firm_col,
        base_profit_col=base_profit_col,
        delta_profit_col=delta_profit_col,
    )
    A = panels["panel_a_increases"].copy()
    B = panels["panel_b_decreases"].copy()

    A["Panel"] = "Panel A. Largest Increases"
    B["Panel"] = "Panel B. Largest Decreases"

    out = pd.concat([A, B], ignore_index=True)

    # Pre-format with leading signs (matching the example)
    out["Δ Profit (millions USD)"] = out["Δ Profit (millions USD)"].map(
        lambda x: "" if pd.isna(x) else f"{float(x):+.{digits_profit}f}"
    )
    out["%Δ"] = out["%Δ"].map(lambda x: "" if pd.isna(x) else f"{float(x):+.{digits_pct}f}")

    return out[["Panel", "Firm", "Δ Profit (millions USD)", "%Δ"]]


def profit_changes_table_latex(
    firm_table: pd.DataFrame,
    *,
    n: int = 5,
    firm_col: str = "firm_ids",
    base_profit_col: str = "pi0_millions_usd",
    delta_profit_col: str = "dpi_millions_usd",
    digits_profit: int = 1,
    digits_pct: int = 1,
) -> str:
    """Booktabs LaTeX that matches the two-panel style in your screenshot."""
    panels = top_bottom_profit_changes(
        firm_table,
        n=n,
        firm_col=firm_col,
        base_profit_col=base_profit_col,
        delta_profit_col=delta_profit_col,
    )
    A = panels["panel_a_increases"]
    B = panels["panel_b_decreases"]

    def fmt(x: float, d: int) -> str:
        return "" if (x is None or (isinstance(x, float) and not np.isfinite(x))) else f"{float(x):+.{d}f}"

    lines = []
    lines.append(r"\\begin{tabular}{lrr}")
    lines.append(r"\\toprule")
    lines.append(r"Firm & $\\Delta$ Profit (millions USD) & \\%$\\Delta$ \\\\")
    lines.append(r"\\midrule")

    lines.append(r"\\multicolumn{3}{l}{\\textit{Panel A. Largest Increases}} \\\\")
    for _, r in A.iterrows():
        lines.append(f"{r['Firm']} & {fmt(r['Δ Profit (millions USD)'], digits_profit)} & {fmt(r['%Δ'], digits_pct)} \\\\")

    lines.append(r"\\\\")
    lines.append(r"\\multicolumn{3}{l}{\\textit{Panel B. Largest Decreases}} \\\\")
    for _, r in B.iterrows():
        lines.append(f"{r['Firm']} & {fmt(r['Δ Profit (millions USD)'], digits_profit)} & {fmt(r['%Δ'], digits_pct)} \\\\")

    lines.append(r"\\bottomrule")
    lines.append(r"\\end{tabular}")
    lines.append(r"\\\\")
    lines.append(r"\\footnotesize Notes: $\\Delta$ is CF minus baseline; \\%$\\Delta$ is the percentage change relative to baseline.")
    return "\n".join(lines)
