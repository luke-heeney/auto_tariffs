from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def latest_saved_output_dir() -> Path:
    base = Path("post_est/saved_outputs")
    dirs = [d for d in base.iterdir() if d.is_dir()]
    if not dirs:
        raise FileNotFoundError("No directories found under post_est/saved_outputs.")
    return max(dirs, key=lambda d: d.stat().st_mtime)


def main() -> None:
    latest = latest_saved_output_dir()
    summary = pd.read_csv(latest / "summary_tbl_all.csv.gz").set_index("Unnamed: 0")
    meta = json.loads((latest / "metadata.json").read_text())
    ev_tbl = pd.read_csv(latest / "ev_tariff_tbl.csv.gz")

    cols = [
        "parts and vehicles tariff (with subsidy)",
        "parts and vehicles tariff (no subsidy)",
        "vehicles-only tariff (with subsidy)",
        "vehicles-only tariff (no subsidy)",
        "no tariff (no subsidy)",
    ]

    def row(name: str):
        return [summary.loc[name, c] for c in cols]

    def esc_pct(v):
        return str(v).replace("%", r"\%")

    def f2(x): return f"{float(x):.2f}"
    def f3(x): return f"{float(x):.3f}"
    def f1(x): return f"{float(x):.1f}"

    dprice = row("Sales-weighted Δ Price (%)")
    markup = row("Sales-weighted Markup (CF, %)")
    prod = row("US Producer Surplus (Δ, billion USD)")
    cs_tot = [esc_pct(v) for v in row("CS Δ total (billion USD)")]
    cs_q1 = [esc_pct(v) for v in row("CS Δ Q1 (billion USD)")]
    cs_q2 = [esc_pct(v) for v in row("CS Δ Q2 (billion USD)")]
    cs_q3 = [esc_pct(v) for v in row("CS Δ Q3 (billion USD)")]
    cs_q4 = [esc_pct(v) for v in row("CS Δ Q4 (billion USD)")]
    cs_q5 = [esc_pct(v) for v in row("CS Δ Q5 (billion USD)")]
    dveh = row("Δ vehicles sold (millions)")
    ev_share = row("EV share of vehicles sold (CF, %)")
    us_share = row("US share of vehicles sold (CF)")
    dus = row("Δ US assembled (millions)")
    tariff = row("Tariff revenue (billion USD)")
    subsidy = row("EV subsidy spending (billion USD)")
    net = row("Net US impact (billion USD)")

    # Update baseline note values from latest metadata + summary baseline column.
    base_subsidy = float(meta["baseline_subsidy_spend_billion_usd"])
    base_row = ev_tbl.loc[ev_tbl["Scenario"] == "no tariff (with subsidy)"].iloc[0]
    base_sales_m = float(base_row["Units (baseline)"]) / 1e6
    base_ev = float(summary.loc["EV share of vehicles sold (CF, %)", "no tariff (with subsidy)"])
    base_us = float(summary.loc["US share of vehicles sold (CF)", "no tariff (with subsidy)"]) * 100

    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Counterfactual Tariff and Subsidy Scenarios: 2024 Market Outcomes}",
        r"\label{tab:cf_summary}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\begin{adjustbox}{center}",
        r"\begin{threeparttable}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r" & \multicolumn{2}{c}{Parts \& vehicles tariff} & \multicolumn{2}{c}{Vehicles-only tariff} & \multicolumn{1}{c}{No tariff} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-6}",
        r" & With subsidy & No subsidy & With subsidy & No subsidy & No subsidy \\",
        r"\midrule",
        rf"$\Delta$ Price (avg, \%)                 & {f2(dprice[0])} & {f2(dprice[1])} & {f2(dprice[2])} & {f2(dprice[3])} & {float(dprice[4]):.4f} \\",
        rf"Markup (avg \%)                         & {f1(markup[0])} & {f1(markup[1])} & {f1(markup[2])} & {f1(markup[3])} & {f1(markup[4])} \\",
        rf"$\Delta$US Producer Surplus (b USD)                  & {f2(prod[0])} & {f2(prod[1])} & {f2(prod[2])} & {f2(prod[3])} & {f2(prod[4])} \\",
        r"\addlinespace[2pt]",
        rf"CS $\Delta$ total (b USD)                & {cs_tot[0]} & {cs_tot[1]} & {cs_tot[2]} & {cs_tot[3]} & {cs_tot[4]} \\",
        rf"CS $\Delta$ Q1 (b USD)                   & {cs_q1[0]} & {cs_q1[1]} & {cs_q1[2]} & {cs_q1[3]} & {cs_q1[4]} \\",
        rf"CS $\Delta$ Q2 (b USD)                   & {cs_q2[0]} & {cs_q2[1]} & {cs_q2[2]} & {cs_q2[3]} & {cs_q2[4]} \\",
        rf"CS $\Delta$ Q3 (b USD)                   & {cs_q3[0]} & {cs_q3[1]} & {cs_q3[2]} & {cs_q3[3]} & {cs_q3[4]} \\",
        rf"CS $\Delta$ Q4 (b USD)                   & {cs_q4[0]} & {cs_q4[1]} & {cs_q4[2]} & {cs_q4[3]} & {cs_q4[4]} \\",
        rf"CS $\Delta$ Q5 (b USD)                   & {cs_q5[0]} & {cs_q5[1]} & {cs_q5[2]} & {cs_q5[3]} & {cs_q5[4]} \\",
        r"\addlinespace[2pt]",
        rf"$\Delta$ vehicles sold (m)               & {f3(dveh[0])} & {f3(dveh[1])} & {f3(dveh[2])} & {f3(dveh[3])} & {f3(dveh[4])} \\",
        rf"EV share (\% sales)                   & {f2(ev_share[0])} & {f2(ev_share[1])} & {f2(ev_share[2])} & {f2(ev_share[3])} & {f2(ev_share[4])} \\",
        rf"US-assembled share (\% sales)                   & {f1(float(us_share[0]) * 100)} & {f1(float(us_share[1]) * 100)} & {f1(float(us_share[2]) * 100)} & {f1(float(us_share[3]) * 100)} & {f1(float(us_share[4]) * 100)} \\",
        rf"$\Delta$ US assembled (m)                & {f3(dus[0])} & {f3(dus[1])} & {f3(dus[2])} & {f3(dus[3])} & {f3(dus[4])} \\",
        rf"Tariff revenue (b USD)                   & {f1(tariff[0])} & {f1(tariff[1])} & {f1(tariff[2])} & {f1(tariff[3])} & {f3(tariff[4])} \\",
        rf"EV subsidy cost (b USD)            & {f2(subsidy[0])} & {f3(subsidy[1])} & {f2(subsidy[2])} & {f3(subsidy[3])} & {f3(subsidy[4])} \\",
        rf"$\Delta$ Net US outcomes (b USD)              & {f2(net[0])}  & {f2(net[1])}  &	{f2(net[2])} &	{f2(net[3])} &	{f2(net[4])} \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}[flushleft]\footnotesize",
        r"\item \textit{Notes:} $\Delta$ entries report counterfactual outcomes relative to the 'no-tarff, with EV subsidies' baseline. Dollars are USD 2015. $\Delta$ Net US outcomes is the change in US producer and consumer surplus, plus tariff revenue, minus (plus) additional EV subsidy expenditure (savings) compared to baseline. US Producer Surplus counts profit changes for US-Headquartered firms. "
        + rf"In the baseline, EV subsidy spending is \${base_subsidy:.2f}b, total vehicle sales are {base_sales_m:.2f} million, EV share is {base_ev:.2f}\%, and US share is {base_us:.1f}\%. "
        + r"Consumer surplus (CS) changes are in billion USD; parentheses report percentage changes. ``With subsidy'' and ``No subsidy'' refer to whether the EV subsidy policy is in place in the counterfactual.",
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{adjustbox}",
        r"\end{table}",
    ]

    out = Path("post_est/outputs/cf_summary_table.tex")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
