from __future__ import annotations

from pathlib import Path

import pandas as pd


PRODUCER_ORDER = [
    "cadillac",
    "chevrolet",
    "ford",
    "hyundai",
    "kia",
    "nissan",
    "tesla",
    "volkswagen",
]

PRODUCER_LABEL = {
    "cadillac": "Cadillac",
    "chevrolet": "Chevrolet",
    "ford": "Ford",
    "hyundai": "Hyundai",
    "kia": "Kia",
    "nissan": "Nissan",
    "tesla": "Tesla",
    "volkswagen": "Volkswagen",
}

YEARS = list(range(2015, 2025))


def _build_base_panel(product_data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(product_data_path)
    df = df[df["ev"] == 1].copy()
    df["subsidy_usd"] = pd.to_numeric(df["subsidy"], errors="coerce") * 100_000.0

    out = (
        df[df["firm_ids"].isin(PRODUCER_ORDER)]
        .groupby(["firm_ids", "market_year"], as_index=False)["subsidy_usd"]
        .mean()
        .pivot(index="firm_ids", columns="market_year", values="subsidy_usd")
        .reindex(index=PRODUCER_ORDER, columns=YEARS)
        .fillna(0.0)
        .round(0)
        .astype(int)
    )
    out.index.name = "Producer"
    return out


def _apply_45w_override(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    for y in (2023, 2024):
        out[y] = out[y].where(out[y] == 7500, 7500)
    return out


def _render_tex(panel: pd.DataFrame, caption: str, label: str) -> str:
    lines: list[str] = []
    lines.append("\\begin{table}[!htbp]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{3pt}")
    lines.append("\\renewcommand{\\arraystretch}{1.15}")
    lines.append("")
    lines.append("\\begin{adjustbox}{center, max width=\\textwidth}")
    lines.append("\\begin{threeparttable}")
    lines.append("\\begin{tabular}{l*{10}{S[table-format=4.0]}}")
    lines.append("\\toprule")
    lines.append(" & \\multicolumn{10}{c}{Market year} \\\\")
    lines.append("\\cmidrule(lr){2-11}")
    lines.append("Producer & {2015} & {2016} & {2017} & {2018} & {2019} & {2020} & {2021} & {2022} & {2023} & {2024} \\\\")
    lines.append("\\midrule")
    for producer in PRODUCER_ORDER:
        row = panel.loc[producer]
        vals = " & ".join(str(int(row[y])) for y in YEARS)
        lines.append(f"{PRODUCER_LABEL[producer]} & {vals} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("")
    lines.append("\\begin{tablenotes}[flushleft]\\footnotesize")
    lines.append("\\item \\textit{Notes:} Entries are average EV subsidy amounts (USD) by producer and market year. Values are rounded to the nearest dollar.")
    lines.append("\\end{tablenotes}")
    lines.append("\\end{threeparttable}")
    lines.append("\\end{adjustbox}")
    lines.append("")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def main() -> None:
    product_data_path = Path("post_est/data/raw/product_data_subsidy.csv")
    out_dir = Path("post_est/outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    base = _build_base_panel(product_data_path)
    base_45w = _apply_45w_override(base)

    base_tex = _render_tex(
        base,
        "Average EV Subsidy Available by Producer",
        "tab:avg_ev_subsidy_by_producer",
    )
    base_45w_tex = _render_tex(
        base_45w,
        "Average EV Subsidy Available by Producer (45W-adjusted in 2023--2024)",
        "tab:avg_ev_subsidy_by_producer_45W",
    )

    p1 = out_dir / "avg_ev_subsidy_by_producer.tex"
    p2 = out_dir / "avg_ev_subsidy_by_producer_45W.tex"
    p1.write_text(base_tex)
    p2.write_text(base_45w_tex)

    print(f"Saved: {p1}")
    print(f"Saved: {p2}")


if __name__ == "__main__":
    main()
