from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


SCENARIOS = {
    "B0": "no tariff (no subsidy)",
    "C1": "parts and vehicles tariff (no subsidy)",
    "C2": "vehicles-only tariff (no subsidy)",
    "C3": "no tariff (with subsidy)",
    "C4": "parts and vehicles tariff (with subsidy)",
    "C5": "vehicles-only tariff (with subsidy)",
}

ROW_ORDER = [
    "Sales-weighted Δ Price (%)",
    "Sales-weighted Markup (CF, %)",
    "US Producer Surplus (Δ, billion USD)",
    "CS Δ total (billion USD)",
    "CS Δ Q1 (billion USD)",
    "CS Δ Q2 (billion USD)",
    "CS Δ Q3 (billion USD)",
    "CS Δ Q4 (billion USD)",
    "CS Δ Q5 (billion USD)",
    "Δ vehicles sold (millions)",
    "EV share of vehicles sold (CF, %)",
    "US share of vehicles sold (CF)",
    "Δ US assembled (millions)",
    "Tariff revenue (billion USD)",
    "EV subsidy spending (billion USD)",
    "Net US impact (billion USD)",
]

CS_ROWS = {
    "CS Δ total (billion USD)",
    "CS Δ Q1 (billion USD)",
    "CS Δ Q2 (billion USD)",
    "CS Δ Q3 (billion USD)",
    "CS Δ Q4 (billion USD)",
    "CS Δ Q5 (billion USD)",
}

DELTA_ROWS = {
    "Sales-weighted Δ Price (%)",
    "US Producer Surplus (Δ, billion USD)",
    "Δ vehicles sold (millions)",
    "Δ US assembled (millions)",
    "Net US impact (billion USD)",
}

_CS_PATTERN = re.compile(r"\s*([+-]?\d+(?:\.\d+)?)\s*\(([-+]?\d+(?:\.\d+)?)%\)\s*")


def latest_saved_output_dir() -> Path:
    base = Path("post_est/saved_outputs")
    dirs = [d for d in base.iterdir() if d.is_dir()]
    if not dirs:
        raise FileNotFoundError("No directories found under post_est/saved_outputs.")
    return max(dirs, key=lambda d: d.stat().st_mtime)


def _parse_float(v: object) -> float:
    if isinstance(v, (int, float, np.floating)):
        return float(v)
    s = str(v).strip()
    if " (" in s:
        s = s.split(" (", 1)[0].strip()
    return float(s)


def _parse_cs(v: object) -> tuple[float, float]:
    s = str(v).strip()
    m = _CS_PATTERN.fullmatch(s)
    if m is None:
        raise ValueError(f"Could not parse CS cell: {v!r}")
    return float(m.group(1)), float(m.group(2))


def _fmt_cs(delta: float, pct: float) -> str:
    if np.isfinite(pct):
        return f"{delta:.3f} ({pct:.1f}%)"
    return f"{delta:.3f} (nan)"


def _clean_zero(x: float, tol: float = 5e-10) -> float:
    return 0.0 if np.isfinite(x) and abs(x) <= tol else x


def build_rebased_summary(summary: pd.DataFrame) -> pd.DataFrame:
    missing = [label for label in SCENARIOS.values() if label not in summary.columns]
    if missing:
        raise KeyError(f"Missing required scenario columns: {missing}")

    out = pd.DataFrame(index=ROW_ORDER, columns=list(SCENARIOS.keys()), dtype=object)
    b0_label = SCENARIOS["B0"]

    for row in ROW_ORDER:
        b0_raw = summary.loc[row, b0_label]

        if row in CS_ROWS:
            d_b0, pct_b0 = _parse_cs(b0_raw)
            if pct_b0 == 0:
                # Degenerate case; should not happen in current pipeline.
                cs_b0_level = np.nan
            else:
                cs_old_baseline = d_b0 / (pct_b0 / 100.0)
                cs_b0_level = cs_old_baseline + d_b0

            for code, label in SCENARIOS.items():
                d_old, _ = _parse_cs(summary.loc[row, label])
                d_new = _clean_zero(d_old - d_b0)
                pct_new = np.nan if (not np.isfinite(cs_b0_level) or cs_b0_level == 0) else 100.0 * d_new / cs_b0_level
                out.loc[row, code] = _fmt_cs(d_new, pct_new)
            continue

        b0_value = _parse_float(b0_raw)
        for code, label in SCENARIOS.items():
            value = _parse_float(summary.loc[row, label])
            if row in DELTA_ROWS:
                value = _clean_zero(value - b0_value)
            out.loc[row, code] = value

    return out


def build_rebased_latex_table(rebased: pd.DataFrame) -> str:
    def esc_pct(v: object) -> str:
        return str(v).replace("%", r"\%")

    def f2(x: object) -> str:
        return f"{float(x):.2f}"

    def f3(x: object) -> str:
        return f"{float(x):.3f}"

    def f1(x: object) -> str:
        return f"{float(x):.1f}"

    dprice = rebased.loc["Sales-weighted Δ Price (%)"]
    markup = rebased.loc["Sales-weighted Markup (CF, %)"]
    prod = rebased.loc["US Producer Surplus (Δ, billion USD)"]
    cs_tot = [esc_pct(v) for v in rebased.loc["CS Δ total (billion USD)"].tolist()]
    cs_q1 = [esc_pct(v) for v in rebased.loc["CS Δ Q1 (billion USD)"].tolist()]
    cs_q2 = [esc_pct(v) for v in rebased.loc["CS Δ Q2 (billion USD)"].tolist()]
    cs_q3 = [esc_pct(v) for v in rebased.loc["CS Δ Q3 (billion USD)"].tolist()]
    cs_q4 = [esc_pct(v) for v in rebased.loc["CS Δ Q4 (billion USD)"].tolist()]
    cs_q5 = [esc_pct(v) for v in rebased.loc["CS Δ Q5 (billion USD)"].tolist()]
    dveh = rebased.loc["Δ vehicles sold (millions)"]
    ev_share = rebased.loc["EV share of vehicles sold (CF, %)"]
    us_share = rebased.loc["US share of vehicles sold (CF)"]
    dus = rebased.loc["Δ US assembled (millions)"]
    tariff = rebased.loc["Tariff revenue (billion USD)"]
    subsidy = rebased.loc["EV subsidy spending (billion USD)"]
    net = rebased.loc["Net US impact (billion USD)"]

    b0_us_share_pct = float(us_share["B0"]) * 100.0

    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Counterfactual Tariff and Subsidy Scenarios: 2024 Market Outcomes}",
        r"\label{tab:cf_summary}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.12}",
        r"\begin{adjustbox}{max width=\textwidth}",
        r"\begin{threeparttable}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r" & \multicolumn{3}{c}{No subsidy} & \multicolumn{3}{c}{With subsidy} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r" & B0: No tariff & C1: Parts+veh tariff & C2: Veh-only tariff & C3: No tariff & C4: Parts+veh tariff & C5: Veh-only tariff \\",
        r"\midrule",
        rf"$\Delta$ Price (avg, \%) & {f2(dprice['B0'])} & {f2(dprice['C1'])} & {f2(dprice['C2'])} & {f2(dprice['C3'])} & {f2(dprice['C4'])} & {f2(dprice['C5'])} \\",
        rf"Markup (avg \%) & {f1(markup['B0'])} & {f1(markup['C1'])} & {f1(markup['C2'])} & {f1(markup['C3'])} & {f1(markup['C4'])} & {f1(markup['C5'])} \\",
        rf"$\Delta$ US Producer Surplus (b USD) & {f2(prod['B0'])} & {f2(prod['C1'])} & {f2(prod['C2'])} & {f2(prod['C3'])} & {f2(prod['C4'])} & {f2(prod['C5'])} \\",
        r"\addlinespace[2pt]",
        rf"CS $\Delta$ total (b USD) & {cs_tot[0]} & {cs_tot[1]} & {cs_tot[2]} & {cs_tot[3]} & {cs_tot[4]} & {cs_tot[5]} \\",
        rf"CS $\Delta$ Q1 (b USD) & {cs_q1[0]} & {cs_q1[1]} & {cs_q1[2]} & {cs_q1[3]} & {cs_q1[4]} & {cs_q1[5]} \\",
        rf"CS $\Delta$ Q2 (b USD) & {cs_q2[0]} & {cs_q2[1]} & {cs_q2[2]} & {cs_q2[3]} & {cs_q2[4]} & {cs_q2[5]} \\",
        rf"CS $\Delta$ Q3 (b USD) & {cs_q3[0]} & {cs_q3[1]} & {cs_q3[2]} & {cs_q3[3]} & {cs_q3[4]} & {cs_q3[5]} \\",
        rf"CS $\Delta$ Q4 (b USD) & {cs_q4[0]} & {cs_q4[1]} & {cs_q4[2]} & {cs_q4[3]} & {cs_q4[4]} & {cs_q4[5]} \\",
        rf"CS $\Delta$ Q5 (b USD) & {cs_q5[0]} & {cs_q5[1]} & {cs_q5[2]} & {cs_q5[3]} & {cs_q5[4]} & {cs_q5[5]} \\",
        r"\addlinespace[2pt]",
        rf"$\Delta$ vehicles sold (m) & {f3(dveh['B0'])} & {f3(dveh['C1'])} & {f3(dveh['C2'])} & {f3(dveh['C3'])} & {f3(dveh['C4'])} & {f3(dveh['C5'])} \\",
        rf"EV share (\% sales) & {f2(ev_share['B0'])} & {f2(ev_share['C1'])} & {f2(ev_share['C2'])} & {f2(ev_share['C3'])} & {f2(ev_share['C4'])} & {f2(ev_share['C5'])} \\",
        rf"US-assembled share (\% sales) & {f1(b0_us_share_pct)} & {f1(float(us_share['C1']) * 100)} & {f1(float(us_share['C2']) * 100)} & {f1(float(us_share['C3']) * 100)} & {f1(float(us_share['C4']) * 100)} & {f1(float(us_share['C5']) * 100)} \\",
        rf"$\Delta$ US assembled (m) & {f3(dus['B0'])} & {f3(dus['C1'])} & {f3(dus['C2'])} & {f3(dus['C3'])} & {f3(dus['C4'])} & {f3(dus['C5'])} \\",
        rf"Tariff revenue (b USD) & {f3(tariff['B0'])} & {f2(tariff['C1'])} & {f2(tariff['C2'])} & {f3(tariff['C3'])} & {f2(tariff['C4'])} & {f2(tariff['C5'])} \\",
        rf"EV subsidy cost (b USD) & {f3(subsidy['B0'])} & {f3(subsidy['C1'])} & {f3(subsidy['C2'])} & {f2(subsidy['C3'])} & {f2(subsidy['C4'])} & {f2(subsidy['C5'])} \\",
        rf"$\Delta$ Net US outcomes (b USD) & {f2(net['B0'])} & {f2(net['C1'])} & {f2(net['C2'])} & {f2(net['C3'])} & {f2(net['C4'])} & {f2(net['C5'])} \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}[flushleft]\footnotesize",
        r"\item \textit{Notes:} $\Delta$ rows are rebased to B0 (no tariff, no subsidy). Levels (markup, EV share, US share, tariff revenue, and subsidy spending) are scenario levels. Dollars are USD 2015.",
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{adjustbox}",
        r"\end{table}",
    ]

    return "\n".join(lines) + "\n"


def main() -> None:
    latest = latest_saved_output_dir()
    summary = pd.read_csv(latest / "summary_tbl_all.csv.gz").set_index("Unnamed: 0")
    summary = summary.reindex(ROW_ORDER)

    rebased = build_rebased_summary(summary)

    # Save machine-readable rebased summary alongside scenario outputs.
    out_saved = latest / "summary_tbl_all_rebased_b0.csv.gz"
    rebased.to_csv(out_saved, compression="gzip")

    # Save a paper-ready LaTeX table in post_est/outputs.
    latex = build_rebased_latex_table(rebased)
    out_tex = Path("post_est/outputs/cf_summary_table_rebased.tex")
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text(latex)

    print(f"Saved rebased summary: {out_saved}")
    print(f"Saved rebased Table 7 LaTeX: {out_tex}")


if __name__ == "__main__":
    main()
