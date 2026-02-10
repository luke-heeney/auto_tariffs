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


def fmt_sig(x: float, sig: int = 3) -> str:
    return f"{float(x):.{sig}g}"


def build_plant_location_table(latest: Path) -> str:
    scen_files = {
        "pv_with": "parts_and_vehicles_tariff__with_subsidy__state_units.csv.gz",
        "vo_with": "vehicles_only_tariff__with_subsidy__state_units.csv.gz",
        "pv_no": "parts_and_vehicles_tariff__no_subsidy__state_units.csv.gz",
        "vo_no": "vehicles_only_tariff__no_subsidy__state_units.csv.gz",
        "nt_no": "no_tariff__no_subsidy__state_units.csv.gz",
    }
    data = {k: pd.read_csv(latest / v).set_index("plant_location") for k, v in scen_files.items()}

    order = [
        "Alabama",
        "Arizona",
        "California",
        "Georgia",
        "Illinois",
        "Indiana",
        "Kansas",
        "Kentucky",
        "Michigan",
        "Mississippi",
        "Missouri",
        "Ohio",
        "South Carolina",
        "Tennessee",
        "Texas",
    ]

    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Counterfactual Changes in Vehicles Sold by Assembly Location (2024)}",
        r"\label{tab:plant_location_changes}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.10}",
        "",
        r"\begin{adjustbox}{center}",
        r"\begin{threeparttable}",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r" &  & \multicolumn{2}{c}{With subsidy} & \multicolumn{3}{c}{No subsidy} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-7}",
        r"Plant location & Measure",
        r"& Parts \& vehicles tariff & Vehicles-only tariff",
        r"& Parts \& vehicles tariff & Vehicles-only tariff & No tariff \\",
        r"\midrule",
    ]

    for loc in order:
        pct_vals = [
            data["pv_with"].loc[loc, "pct_change"],
            data["vo_with"].loc[loc, "pct_change"],
            data["pv_no"].loc[loc, "pct_change"],
            data["vo_no"].loc[loc, "pct_change"],
            data["nt_no"].loc[loc, "pct_change"],
        ]
        delta_100k = [
            data["pv_with"].loc[loc, "delta_units"] / 100000.0,
            data["vo_with"].loc[loc, "delta_units"] / 100000.0,
            data["pv_no"].loc[loc, "delta_units"] / 100000.0,
            data["vo_no"].loc[loc, "delta_units"] / 100000.0,
            data["nt_no"].loc[loc, "delta_units"] / 100000.0,
        ]
        lines.append(
            f"{loc} & \\% change             & "
            + " & ".join(f"{fmt_sig(v)}\\%" for v in pct_vals)
            + r" \\"
        )
        lines.append(
            r"        & $\Delta$ units (100k) & "
            + " & ".join(fmt_sig(v) for v in delta_100k)
            + r" \\"
        )
        lines.append(r"\addlinespace[2pt]")

    us_vals = [
        data["pv_with"].loc["United States", "delta_units"] / 100000.0,
        data["vo_with"].loc["United States", "delta_units"] / 100000.0,
        data["pv_no"].loc["United States", "delta_units"] / 100000.0,
        data["vo_no"].loc["United States", "delta_units"] / 100000.0,
        data["nt_no"].loc["United States", "delta_units"] / 100000.0,
    ]

    lines.extend(
        [
            r"\midrule",
            r"United States & \% change             &  &  &  &  &  \\",
            r"              & $\Delta$ units (100k) & " + " & ".join(fmt_sig(v) for v in us_vals) + r" \\",
            r"\bottomrule",
            r"\end{tabular}",
            "",
            r"\begin{tablenotes}[flushleft]\footnotesize",
            r"\item \textit{Notes:} For each plant location, the first row reports the percent change in units sold and the second row reports the change in units sold in hundreds of thousands of vehicles (100k), relative to the corresponding baseline scenario. The United States row reports the sum of state-level changes (percent changes omitted).",
            r"\end{tablenotes}",
            r"\end{threeparttable}",
            r"\end{adjustbox}",
            "",
            r"\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def build_state_cs_table(latest: Path) -> str:
    scen_files = {
        "pv_with": "parts_and_vehicles_tariff__with_subsidy__state_cs.csv.gz",
        "vo_with": "vehicles_only_tariff__with_subsidy__state_cs.csv.gz",
        "pv_no": "parts_and_vehicles_tariff__no_subsidy__state_cs.csv.gz",
        "vo_no": "vehicles_only_tariff__no_subsidy__state_cs.csv.gz",
        "nt_no": "no_tariff__no_subsidy__state_cs.csv.gz",
    }
    data = {k: pd.read_csv(latest / v).set_index("state") for k, v in scen_files.items()}
    meta = json.loads((latest / "metadata.json").read_text())
    factor_billion = float(meta["total_market_size"]) * float(meta["price_scale_usd_per_unit"]) / 1e9

    order = [
        ("ALABAMA", "AL"), ("ALASKA", "AK"), ("ARIZONA", "AZ"), ("ARKANSAS", "AR"), ("CALIFORNIA", "CA"),
        ("COLORADO", "CO"), ("CONNECTICUT", "CT"), ("DELAWARE", "DE"), ("DISTRICT OF COLUMBIA", "DC"),
        ("FLORIDA", "FL"), ("GEORGIA", "GA"), ("HAWAII", "HI"), ("IDAHO", "ID"), ("ILLINOIS", "IL"),
        ("INDIANA", "IN"), ("IOWA", "IA"), ("KANSAS", "KS"), ("KENTUCKY", "KY"), ("LOUISIANA", "LA"),
        ("MAINE", "ME"), ("MARYLAND", "MD"), ("MASSACHUSETTS", "MA"), ("MICHIGAN", "MI"), ("MINNESOTA", "MN"),
        ("MISSISSIPPI", "MS"), ("MISSOURI", "MO"), ("MONTANA", "MT"), ("NEBRASKA", "NE"), ("NEVADA", "NV"),
        ("NEW HAMPSHIRE", "NH"), ("NEW JERSEY", "NJ"), ("NEW MEXICO", "NM"), ("NEW YORK", "NY"),
        ("NORTH CAROLINA", "NC"), ("NORTH DAKOTA", "ND"), ("OHIO", "OH"), ("OKLAHOMA", "OK"), ("OREGON", "OR"),
        ("PENNSYLVANIA", "PA"), ("RHODE ISLAND", "RI"), ("SOUTH CAROLINA", "SC"), ("SOUTH DAKOTA", "SD"),
        ("TENNESSEE", "TN"), ("TEXAS", "TX"), ("UTAH", "UT"), ("VERMONT", "VT"), ("VIRGINIA", "VA"),
        ("WASHINGTON", "WA"), ("WEST VIRGINIA", "WV"), ("WISCONSIN", "WI"), ("WYOMING", "WY"),
    ]

    def entry(df: pd.DataFrame, state: str) -> str:
        dcs_b = float(df.loc[state, "dCS"]) * factor_billion
        pct = float(df.loc[state, "pct_change_vs_baseline"])
        return f"{fmt_sig(dcs_b)} ({fmt_sig(pct)}\\%)"

    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Counterfactual Changes in Consumer Surplus by State (2024)}",
        r"\label{tab:cs_by_state}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\renewcommand{\arraystretch}{0.92}",
        "",
        r"\begin{adjustbox}{center, max width=\textwidth}",
        r"\begin{threeparttable}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r" & \multicolumn{2}{c}{With subsidy} & \multicolumn{3}{c}{No subsidy} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-6}",
        r"State",
        r"& Parts \& vehicles tariff & Vehicles-only tariff",
        r"& Parts \& vehicles tariff & Vehicles-only tariff & No tariff \\",
        r"\midrule",
    ]

    for sname, sabbr in order:
        vals = [
            entry(data["pv_with"], sname),
            entry(data["vo_with"], sname),
            entry(data["pv_no"], sname),
            entry(data["vo_no"], sname),
            entry(data["nt_no"], sname),
        ]
        lines.append(f"{sabbr} & " + " & ".join(vals) + r" \\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            "",
            r"\begin{tablenotes}[flushleft]\footnotesize",
            r"\item \textit{Notes:} Each entry reports the change in consumer surplus in billion USD, with the percent change shown in parentheses. Values are relative to the corresponding baseline.",
            r"\end{tablenotes}",
            r"\end{threeparttable}",
            r"\end{adjustbox}",
            "",
            r"\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    latest = latest_saved_output_dir()
    out_dir = Path("post_est/outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    plant_tex = build_plant_location_table(latest)
    state_cs_tex = build_state_cs_table(latest)

    (out_dir / "plant_location_changes.tex").write_text(plant_tex)
    (out_dir / "cs_by_state.tex").write_text(state_cs_tex)

    print(f"Saved: {out_dir / 'plant_location_changes.tex'}")
    print(f"Saved: {out_dir / 'cs_by_state.tex'}")


if __name__ == "__main__":
    main()
