from __future__ import annotations

from pathlib import Path

import pandas as pd


OUT_DIR = Path("cost_side/outputs")


def fmt(value: float) -> str:
    return f"{value:.3f}"


def coef_cell(row: pd.Series) -> str:
    return f"{fmt(float(row['estimate']))} ({fmt(float(row['std_error']))})"


def select_row(
    df: pd.DataFrame,
    *,
    sample: str,
    spec: str,
    coefficient: str | None = None,
    contains: str | None = None,
) -> pd.Series:
    mask = (df["sample"] == sample) & (df["spec"] == spec)
    if coefficient is not None:
        mask &= df["coefficient"] == coefficient
    if contains is not None:
        mask &= df["coefficient"].astype(str).str.contains(contains, regex=False)
    rows = df.loc[mask]
    if len(rows) != 1:
        raise ValueError(
            f"Expected exactly one row for sample={sample}, spec={spec}, "
            f"coefficient={coefficient}, contains={contains}; found {len(rows)}"
        )
    return rows.iloc[0]


def write(path: Path, body: str) -> None:
    path.write_text(body.strip() + "\n")


def table_note(text: str, ncols: int) -> str:
    return (
        f"\\multicolumn{{{ncols}}}{{p{{0.94\\textwidth}}}}"
        f"{{\\footnotesize Notes: {text}}} \\\\"
    )


def write_registry_table() -> None:
    placebo = pd.read_csv(OUT_DIR / "cost_reg_placebo_coefficients.csv")
    robust = pd.read_csv(OUT_DIR / "cost_reg_robustness_coefficients.csv")
    decomp = pd.read_csv(OUT_DIR / "cost_reg_price_markup_decomp_coefficients.csv")
    loo = pd.read_csv(OUT_DIR / "leave_one_country_out_coefficients.csv", keep_default_na=False)
    alt = pd.read_csv(OUT_DIR / "cost_reg_alt_exposure_coefficients.csv")

    baseline_l = select_row(
        placebo,
        sample="domestic_current_rer_levels",
        spec="levels",
        coefficient="ln_inv_rer_code1:pcOth1_pct1_lag1",
    )
    baseline_fd = select_row(
        placebo,
        sample="domestic_current_rer_fd",
        spec="fd",
        coefficient="ln_inv_rer_code1:pcOth1_pct1_lag1",
    )
    foreign_l = select_row(
        placebo,
        sample="foreign_current_rer_levels",
        spec="levels",
        coefficient="ln_inv_rer_code1:pcOth1_pct1_lag1",
    )
    foreign_fd = select_row(
        placebo,
        sample="foreign_current_rer_fd",
        spec="fd",
        coefficient="ln_inv_rer_code1:pcOth1_pct1_lag1",
    )
    joint_future_l = select_row(
        placebo,
        sample="domestic_current_plus_future_rer_levels",
        spec="levels",
        contains="tplus1",
    )
    joint_future_fd = select_row(
        placebo,
        sample="domestic_current_plus_future_rer_fd",
        spec="fd",
        contains="tplus1",
    )
    full_loo_l = select_row(
        loo,
        sample="full_sample",
        spec="levels",
        coefficient="ln_inv_rer_code1:pcOth1_pct1_lag1",
    )
    full_loo_fd = select_row(
        loo,
        sample="full_sample",
        spec="fd",
        coefficient="ln_inv_rer_code1:pcOth1_pct1_lag1",
    )
    loo_levels = loo.loc[
        (loo["spec"] == "levels")
        & (loo["sample"] != "full_sample")
        & (loo["coefficient"] == "ln_inv_rer_code1:pcOth1_pct1_lag1")
    ].copy()
    loo_levels["abs_shift"] = (loo_levels["estimate"] - float(full_loo_l["estimate"])).abs()
    largest_loo_l = loo_levels.sort_values("abs_shift", ascending=False).iloc[0] if not loo_levels.empty else full_loo_l
    largest_loo_country = str(largest_loo_l.get("omitted_country", "None"))
    largest_loo_fd = select_row(
        loo,
        sample=str(largest_loo_l["sample"]),
        spec="fd",
        coefficient="ln_inv_rer_code1:pcOth1_pct1_lag1",
    )
    rer_main_l = select_row(
        robust,
        sample="canonical_domestic",
        spec="levels_rer_main",
        coefficient="ln_inv_rer_code1:pcOth1_pct1_lag1",
    )
    rer_main_fd = select_row(
        robust,
        sample="canonical_domestic",
        spec="fd_rer_main",
        coefficient="ln_inv_rer_code1:pcOth1_pct1_lag1",
    )
    observed_total_fd = select_row(
        alt,
        sample="observed_total_share",
        spec="fd_current",
        coefficient="ln_inv_rer_code1:observed_foreign_share_lag1",
    )
    observed_total_forward_fd = select_row(
        alt,
        sample="observed_total_share",
        spec="fd_forward",
        coefficient="d_ln_inv_rer_code1_tplus1:observed_foreign_share_lag1",
    )
    cost_l = select_row(
        decomp,
        sample="recovered_cost",
        spec="levels_current",
        coefficient="ln_inv_rer_code1:pcOth1_pct1_lag1",
    )
    price_l = select_row(
        decomp,
        sample="observed_price",
        spec="levels_current",
        coefficient="ln_inv_rer_code1:pcOth1_pct1_lag1",
    )
    markup_l = select_row(
        decomp,
        sample="recovered_markup",
        spec="levels_current",
        coefficient="ln_inv_rer_code1:pcOth1_pct1_lag1",
    )

    rows = [
        (
            "Current RER baseline",
            "Does current exposure-weighted RER move recovered costs?",
            f"Levels {coef_cell(baseline_l)}; FD {coef_cell(baseline_fd)}",
            "Supportive",
            "Cost channel is visible in the baseline sample.",
        ),
        (
            "Foreign-assembled placebo",
            "Does the same pattern appear where the U.S. parts-cost channel should be weak?",
            f"Levels {coef_cell(foreign_l)}; FD {coef_cell(foreign_fd)}",
            "Passes",
            "Foreign placebo is close to zero.",
        ),
        (
            "Conditional future-RER placebo",
            "Does future RER predict current costs after controlling for current RER?",
            f"Future term: levels {coef_cell(joint_future_l)}; FD {coef_cell(joint_future_fd)}",
            "Concern",
            "Not a clean timing design.",
        ),
        (
            "Leave-one-country-out",
            "Is the baseline sign driven by one primary source country?",
            f"Full levels {coef_cell(full_loo_l)}; largest shift drops {largest_loo_country}: levels {coef_cell(largest_loo_l)}; FD {coef_cell(largest_loo_fd)}",
            "Mixed",
            "Sign stability depends on which source countries carry the identifying variation.",
        ),
        (
            "Alternative exposure definitions",
            "Does the FD result depend on using only the primary foreign-parts share?",
            f"Observed-total FD {coef_cell(observed_total_fd)}; forward FD {coef_cell(observed_total_forward_fd)}",
            "Mixed",
            "Current estimate is stable, but the forward placebo remains positive.",
        ),
        (
            "RER main effect",
            "Does the exposure interaction survive a plain RER term?",
            f"Interaction: levels {coef_cell(rer_main_l)}; FD {coef_cell(rer_main_fd)}",
            "Supportive",
            "The exposure term remains positive and larger.",
        ),
        (
            "Price/markup decomposition",
            "Does the term load on costs or recovered markups?",
            f"Cost {coef_cell(cost_l)}; price {coef_cell(price_l)}; markup {coef_cell(markup_l)}",
            "Supportive diagnostic",
            "The term is not a markup-only pattern.",
        ),
    ]

    body_rows = "\n".join(
        f"{test} & {question} & {estimate} & {status} & {implication} \\\\"
        for test, question, estimate, status, implication in rows
    )
    tex = rf"""
\begin{{table}}[htbp]
\centering
\caption{{Robustness test registry}}
\label{{tab:cost_side_note_registry}}
\footnotesize
\begin{{tabular}}{{p{{0.14\textwidth}}p{{0.22\textwidth}}p{{0.18\textwidth}}p{{0.10\textwidth}}p{{0.17\textwidth}}}}
\toprule
Test & Question & Main estimate & Status & Implication \\
\midrule
{body_rows}
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    write(OUT_DIR / "cost_side_note_registry_table.tex", tex)


def write_sample_table() -> None:
    panel = pd.read_csv(OUT_DIR / "panel_build_sample_counts.csv")
    regression = pd.read_csv(OUT_DIR / "cost_reg_placebo_sample_counts.csv")

    def panel_row(sample: str, label: str, note: str) -> tuple[str, int, int, str]:
        row = panel.loc[panel["sample"] == sample]
        if len(row) != 1:
            raise ValueError(f"Expected one panel sample row for {sample}, found {len(row)}")
        row = row.iloc[0]
        return label, int(row["rows"]), int(row["make_models"]), note

    def regression_row(sample: str, label: str, note: str) -> tuple[str, int, int, str]:
        row = regression.loc[regression["sample"] == sample]
        if len(row) != 1:
            raise ValueError(f"Expected one regression sample row for {sample}, found {len(row)}")
        row = row.iloc[0]
        return label, int(row["rows"]), int(row["make_models"]), note

    rows = [
        panel_row(
            "all",
            "All matched vehicles",
            "Matched to sourcing, recovered costs, characteristics, and a primary-source RER.",
        ),
        panel_row(
            "us_all",
            "U.S.-assembled",
            "Domestic assembly sample before canonical exposure restrictions.",
        ),
        panel_row(
            "canonical_domestic",
            "Canonical domestic panel",
            "U.S.-assembled vehicles with a stable primary source country and no raw primary-country conflict within product-year.",
        ),
        panel_row(
            "foreign_source_stable",
            "Foreign placebo panel",
            "Foreign-assembled vehicles with stable primary source country.",
        ),
        regression_row(
            "domestic_current_rer_levels",
            "Baseline levels regression",
            "Requires lagged exposure, controls, and make-model/year fixed effects.",
        ),
        regression_row(
            "domestic_current_rer_fd",
            "Baseline FD regression",
            "Requires consecutive-year observations and differenced controls.",
        ),
    ]

    body_rows = "\n".join(
        f"{label} & {rows_count} & {make_models} & {note} \\\\"
        for label, rows_count, make_models, note in rows
    )
    tex = rf"""
\begin{{table}}[htbp]
\centering
\caption{{Cost-side samples and filters}}
\label{{tab:cost_side_note_samples}}
\footnotesize
\begin{{tabular}}{{p{{0.24\textwidth}}rrp{{0.43\textwidth}}}}
\toprule
Sample & Rows & Make-models & Definition \\
\midrule
{body_rows}
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    write(OUT_DIR / "cost_side_note_sample_table.tex", tex)


def write_timing_table() -> None:
    df = pd.read_csv(OUT_DIR / "cost_reg_placebo_coefficients.csv")

    rows = [
        (
            "Current RER baseline",
            select_row(
                df,
                sample="domestic_current_rer_levels",
                spec="levels",
                coefficient="ln_inv_rer_code1:pcOth1_pct1_lag1",
            ),
            select_row(
                df,
                sample="domestic_current_rer_fd",
                spec="fd",
                coefficient="ln_inv_rer_code1:pcOth1_pct1_lag1",
            ),
            "Main channel estimate.",
        ),
        (
            "Future RER only",
            select_row(
                df,
                sample="domestic_future_rer_levels",
                spec="levels",
                coefficient="ln_inv_rer_code1_tplus1:pcOth1_pct1_lag1",
            ),
            select_row(
                df,
                sample="domestic_future_rer_fd",
                spec="fd",
                coefficient="d_ln_inv_rer_code1_tplus1:pcOth1_pct1_lag1",
            ),
            "Standalone timing placebo.",
        ),
        (
            "Current plus future: current term",
            select_row(
                df,
                sample="domestic_current_plus_future_rer_levels",
                spec="levels",
                coefficient="ln_inv_rer_code1:pcOth1_pct1_lag1",
            ),
            select_row(
                df,
                sample="domestic_current_plus_future_rer_fd",
                spec="fd",
                coefficient="ln_inv_rer_code1:pcOth1_pct1_lag1",
            ),
            "Current term conditional on future RER.",
        ),
        (
            "Current plus future: future term",
            select_row(
                df,
                sample="domestic_current_plus_future_rer_levels",
                spec="levels",
                contains="tplus1",
            ),
            select_row(
                df,
                sample="domestic_current_plus_future_rer_fd",
                spec="fd",
                contains="tplus1",
            ),
            "Main timing diagnostic.",
        ),
        (
            "Foreign-assembled placebo",
            select_row(
                df,
                sample="foreign_current_rer_levels",
                spec="levels",
                coefficient="ln_inv_rer_code1:pcOth1_pct1_lag1",
            ),
            select_row(
                df,
                sample="foreign_current_rer_fd",
                spec="fd",
                coefficient="ln_inv_rer_code1:pcOth1_pct1_lag1",
            ),
            "Placebo sample.",
        ),
    ]
    body_rows = "\n".join(
        f"{label} & {coef_cell(levels)} & {int(levels['nobs'])} & "
        f"{coef_cell(fd)} & {int(fd['nobs'])} & {takeaway} \\\\"
        for label, levels, fd, takeaway in rows
    )
    tex = rf"""
\begin{{table}}[htbp]
\centering
\caption{{Timing and placebo regressions}}
\label{{tab:cost_side_note_timing}}
\footnotesize
\begin{{tabular}}{{p{{0.20\textwidth}}ccccp{{0.20\textwidth}}}}
\toprule
Specification & Levels coef. & Levels $N$ & FD coef. & FD $N$ & Takeaway \\
\midrule
{body_rows}
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    write(OUT_DIR / "cost_side_note_timing_table.tex", tex)


def write_exposure_table() -> None:
    loo = pd.read_csv(OUT_DIR / "leave_one_country_out_coefficients.csv", keep_default_na=False)
    counts_path = OUT_DIR / "leave_one_country_out_sample_counts.csv"
    counts = pd.read_csv(counts_path, keep_default_na=False) if counts_path.exists() else pd.DataFrame()

    levels = loo.loc[
        (loo["spec"] == "levels")
        & (loo["coefficient"] == "ln_inv_rer_code1:pcOth1_pct1_lag1")
    ].copy()
    if not counts.empty:
        levels = levels.merge(
            counts[["spec", "omitted_rows"]],
            left_on="sample",
            right_on="spec",
            how="left",
            suffixes=("", "_count"),
        )
    else:
        levels["omitted_rows"] = 0
    levels["sort_full"] = levels["sample"].ne("full_sample").astype(int)
    levels["sort_rows"] = levels["omitted_rows"].fillna(0).astype(float)
    levels = levels.sort_values(["sort_full", "sort_rows", "omitted_country"], ascending=[True, False, True])

    rows = []
    for _, level_row in levels.iterrows():
        label = str(level_row["omitted_country"])
        if label == "None":
            label = "Full sample"
        sample = str(level_row["sample"])
        fd_row = select_row(
            loo,
            sample=sample,
            spec="fd",
            coefficient="ln_inv_rer_code1:pcOth1_pct1_lag1",
        )
        if sample == "full_sample":
            takeaway = "Reference sample."
        elif float(level_row["estimate"]) > 0 and float(fd_row["estimate"]) > 0:
            takeaway = "Sign remains positive."
        else:
            takeaway = "Sign is sensitive in this omission."
        rows.append((label, level_row, fd_row, takeaway))

    def maybe_cell(row: pd.Series | None) -> str:
        return "--" if row is None else f"{coef_cell(row)}; $N={int(row['nobs'])}$"

    body_rows = "\n".join(
        f"{label} & {maybe_cell(levels)} & {maybe_cell(fd)} & {takeaway} \\\\"
        for label, levels, fd, takeaway in rows
    )
    tex = rf"""
\begin{{table}}[htbp]
\centering
\caption{{Leave-one-country-out exposure checks}}
\label{{tab:cost_side_note_exposure}}
\footnotesize
\begin{{tabular}}{{p{{0.20\textwidth}}p{{0.20\textwidth}}p{{0.20\textwidth}}p{{0.24\textwidth}}}}
\toprule
Omitted country & Levels & First differences & Takeaway \\
\midrule
{body_rows}
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    write(OUT_DIR / "cost_side_note_exposure_table.tex", tex)


def write_alt_exposure_table() -> None:
    df = pd.read_csv(OUT_DIR / "cost_reg_alt_exposure_coefficients.csv")

    primary_current = select_row(
        df,
        sample="primary_share",
        spec="fd_current",
        coefficient="ln_inv_rer_code1:pcOth1_pct1_lag1",
    )
    primary_forward = select_row(
        df,
        sample="primary_share",
        spec="fd_forward",
        coefficient="d_ln_inv_rer_code1_tplus1:pcOth1_pct1_lag1",
    )
    observed_current = select_row(
        df,
        sample="observed_total_share",
        spec="fd_current",
        coefficient="ln_inv_rer_code1:observed_foreign_share_lag1",
    )
    observed_forward = select_row(
        df,
        sample="observed_total_share",
        spec="fd_forward",
        coefficient="d_ln_inv_rer_code1_tplus1:observed_foreign_share_lag1",
    )
    high_current_main = select_row(
        df,
        sample="high_exposure_indicator",
        spec="fd_current",
        coefficient="ln_inv_rer_code1",
    )
    high_current_diff = select_row(
        df,
        sample="high_exposure_indicator",
        spec="fd_current",
        coefficient="ln_inv_rer_code1:high_exposure_q4_lag1",
    )
    high_forward_main = select_row(
        df,
        sample="high_exposure_indicator",
        spec="fd_forward",
        coefficient="d_ln_inv_rer_code1_tplus1",
    )
    high_forward_diff = select_row(
        df,
        sample="high_exposure_indicator",
        spec="fd_forward",
        coefficient="d_ln_inv_rer_code1_tplus1:high_exposure_q4_lag1",
    )

    rows = [
        (
            "Primary foreign share",
            coef_cell(primary_current),
            coef_cell(primary_forward),
            "Baseline FD exposure.",
        ),
        (
            "Observed foreign share in first content block",
            coef_cell(observed_current),
            coef_cell(observed_forward),
            "Current estimate is nearly unchanged; forward placebo remains similar.",
        ),
        (
            "High-exposure indicator: low-exposure term",
            coef_cell(high_current_main),
            coef_cell(high_forward_main),
            "Low-exposure products show weak current response but a positive forward term.",
        ),
        (
            "High-exposure indicator: high-exposure differential",
            coef_cell(high_current_diff),
            coef_cell(high_forward_diff),
            "High-exposure differential is positive for current shocks and not positive for forward shocks.",
        ),
    ]

    body_rows = "\n".join(
        f"{label} & {current} & {forward} & {takeaway} \\\\"
        for label, current, forward, takeaway in rows
    )
    tex = rf"""
\begin{{table}}[htbp]
\centering
\caption{{Alternative exposure definitions, first-difference regressions}}
\label{{tab:cost_side_note_alt_exposure}}
\footnotesize
\begin{{tabular}}{{p{{0.28\textwidth}}p{{0.18\textwidth}}p{{0.18\textwidth}}p{{0.27\textwidth}}}}
\toprule
Exposure definition & Current RER & Forward RER & Takeaway \\
\midrule
{body_rows}
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    write(OUT_DIR / "cost_side_note_alt_exposure_table.tex", tex)


def write_vehicle_type_table() -> None:
    df = pd.read_csv(OUT_DIR / "cost_reg_vehicle_type_coefficients.csv")

    def get(sample: str, coefficient: str) -> pd.Series:
        rows = df.loc[(df["sample"] == sample) & (df["coefficient"] == coefficient)]
        if len(rows) != 1:
            raise ValueError(f"Expected one vehicle-type row for {sample}/{coefficient}, found {len(rows)}")
        return rows.iloc[0]

    types = [
        ("car", "Car"),
        ("truck", "Truck"),
        ("suv", "SUV"),
        ("van", "Van"),
    ]
    rows = []
    for suffix, label in types:
        rows.append(
            (
                label,
                coef_cell(get("levels_controls", f"rho_rer_{suffix}")),
                coef_cell(get("levels_no_controls", f"rho_rer_{suffix}")),
                coef_cell(get("fd_controls", f"rho_rer_fd_{suffix}")),
                coef_cell(get("fd_no_controls", f"rho_rer_fd_{suffix}")),
            )
        )

    body_rows = "\n".join(
        f"{label} & {levels_controls} & {levels_no_controls} & "
        f"{fd_controls} & {fd_no_controls} \\\\"
        for label, levels_controls, levels_no_controls, fd_controls, fd_no_controls in rows
    )
    tex = rf"""
\begin{{table}}[htbp]
\centering
\caption{{Vehicle-type-specific pass-through diagnostics}}
\label{{tab:cost_side_note_vehicle_type}}
\footnotesize
\begin{{tabular}}{{lcccc}}
\toprule
Vehicle type & Levels, controls & Levels, no controls & FD, controls & FD, no controls \\
\midrule
{body_rows}
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    write(OUT_DIR / "cost_side_note_vehicle_type_table.tex", tex)


def write_decomposition_table() -> None:
    df = pd.read_csv(OUT_DIR / "cost_reg_price_markup_decomp_coefficients.csv")
    rows = []
    for sample, label in [
        ("recovered_cost", "Recovered marginal cost"),
        ("observed_price", "Observed price"),
        ("recovered_markup", "Recovered markup"),
    ]:
        rows.append(
            (
                label,
                select_row(
                    df,
                    sample=sample,
                    spec="levels_current",
                    coefficient="ln_inv_rer_code1:pcOth1_pct1_lag1",
                ),
                select_row(
                    df,
                    sample=sample,
                    spec="fd_current",
                    coefficient="ln_inv_rer_code1:pcOth1_pct1_lag1",
                ),
                select_row(
                    df,
                    sample=sample,
                    spec="levels_forward",
                    coefficient="ln_inv_rer_code1_tplus1:pcOth1_pct1_lag1",
                ),
                select_row(
                    df,
                    sample=sample,
                    spec="fd_forward",
                    coefficient="d_ln_inv_rer_code1_tplus1:pcOth1_pct1_lag1",
                ),
            )
        )
    body_rows = "\n".join(
        f"{label} & {coef_cell(cur_l)} & {coef_cell(cur_fd)} & "
        f"{coef_cell(fwd_l)} & {coef_cell(fwd_fd)} \\\\"
        for label, cur_l, cur_fd, fwd_l, fwd_fd in rows
    )
    tex = rf"""
\begin{{table}}[htbp]
\centering
\caption{{Price and markup decomposition}}
\label{{tab:cost_side_note_decomp}}
\footnotesize
\begin{{tabular}}{{lcccc}}
\toprule
Outcome & Current levels & Current FD & Forward levels & Forward FD \\
\midrule
{body_rows}
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    write(OUT_DIR / "cost_side_note_decomp_table.tex", tex)


def main() -> None:
    write_sample_table()
    write_registry_table()
    write_timing_table()
    write_exposure_table()
    write_alt_exposure_table()
    write_vehicle_type_table()
    write_decomposition_table()
    print("Saved:")
    for name in [
        "cost_side_note_sample_table.tex",
        "cost_side_note_registry_table.tex",
        "cost_side_note_timing_table.tex",
        "cost_side_note_exposure_table.tex",
        "cost_side_note_alt_exposure_table.tex",
        "cost_side_note_vehicle_type_table.tex",
        "cost_side_note_decomp_table.tex",
    ]:
        print(" -", OUT_DIR / name)


if __name__ == "__main__":
    main()
