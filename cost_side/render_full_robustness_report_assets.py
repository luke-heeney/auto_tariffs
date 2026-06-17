from __future__ import annotations

from pathlib import Path

import pandas as pd


OUT_DIR = Path("cost_side/outputs")


def fmt_num(value: float) -> str:
    return f"{value:.3f}"


def coef_cell(value: float, se: float) -> str:
    return f"{fmt_num(value)} ({fmt_num(se)})"


def load_placebo_rows() -> dict[tuple[str, str], dict[str, float]]:
    df = pd.read_csv(OUT_DIR / "cost_reg_placebo_coefficients.csv")
    rows: dict[tuple[str, str], dict[str, float]] = {}
    for _, row in df.iterrows():
      rows[(row["sample"], row["spec"])] = row.to_dict()
    return rows


def write_main_identification_table() -> None:
    rows = load_placebo_rows()

    lvl_current = rows[("domestic_current_rer_levels", "levels")]
    lvl_future = rows[("domestic_future_rer_levels", "levels")]
    lvl_joint_current = rows[("domestic_current_plus_future_rer_levels", "levels")]
    lvl_joint_future = rows[("domestic_current_plus_future_rer_levels", "levels")]
    lvl_foreign = rows[("foreign_current_rer_levels", "levels")]

    fd_current = rows[("domestic_current_rer_fd", "fd")]
    fd_future = rows[("domestic_future_rer_fd", "fd")]
    fd_joint_current = rows[("domestic_current_plus_future_rer_fd", "fd")]
    fd_joint_future = rows[("domestic_current_plus_future_rer_fd", "fd")]
    fd_foreign = rows[("foreign_current_rer_fd", "fd")]

    # The joint specs appear twice in the CSV, once for each coefficient.
    placebo_df = pd.read_csv(OUT_DIR / "cost_reg_placebo_coefficients.csv")

    def select(sample: str, spec: str, contains: str) -> tuple[float, float, int]:
        row = placebo_df[
            (placebo_df["sample"] == sample)
            & (placebo_df["spec"] == spec)
            & (placebo_df["coefficient"].str.contains(contains, regex=False))
        ].iloc[0]
        return float(row["estimate"]), float(row["std_error"]), int(row["nobs"])

    lvl_joint_current_est, lvl_joint_current_se, lvl_joint_n = select(
        "domestic_current_plus_future_rer_levels", "levels", "ln_inv_rer_code1"
    )
    lvl_joint_future_est, lvl_joint_future_se, _ = select(
        "domestic_current_plus_future_rer_levels", "levels", "tplus1"
    )
    fd_joint_current_est, fd_joint_current_se, fd_joint_n = select(
        "domestic_current_plus_future_rer_fd", "fd", "ln_inv_rer_code1"
    )
    fd_joint_future_est, fd_joint_future_se, _ = select(
        "domestic_current_plus_future_rer_fd", "fd", "tplus1"
    )

    tex = f"""
\\begin{{table}}[htbp]
\\centering
\\caption{{Main exchange-rate identification results}}
\\label{{tab:full_robustness_main}}
\\begin{{tabular}}{{lcccc}}
\\toprule
 & Baseline current & Forward only & Current + forward & Foreign placebo \\\\
\\midrule
\\multicolumn{{5}}{{l}}{{\\textit{{Panel A: Levels}}}} \\\\
Current RER exposure term & {coef_cell(float(lvl_current["estimate"]), float(lvl_current["std_error"]))} &  & {coef_cell(lvl_joint_current_est, lvl_joint_current_se)} & {coef_cell(float(lvl_foreign["estimate"]), float(lvl_foreign["std_error"]))} \\\\
Forward RER exposure term &  & {coef_cell(float(lvl_future["estimate"]), float(lvl_future["std_error"]))} & {coef_cell(lvl_joint_future_est, lvl_joint_future_se)} &  \\\\
Observations & {int(lvl_current["nobs"])} & {int(lvl_future["nobs"])} & {lvl_joint_n} & {int(lvl_foreign["nobs"])} \\\\
\\addlinespace
\\multicolumn{{5}}{{l}}{{\\textit{{Panel B: First differences}}}} \\\\
Current RER exposure term & {coef_cell(float(fd_current["estimate"]), float(fd_current["std_error"]))} &  & {coef_cell(fd_joint_current_est, fd_joint_current_se)} & {coef_cell(float(fd_foreign["estimate"]), float(fd_foreign["std_error"]))} \\\\
Forward RER exposure term &  & {coef_cell(float(fd_future["estimate"]), float(fd_future["std_error"]))} & {coef_cell(fd_joint_future_est, fd_joint_future_se)} &  \\\\
Observations & {int(fd_current["nobs"])} & {int(fd_future["nobs"])} & {fd_joint_n} & {int(fd_foreign["nobs"])} \\\\
\\bottomrule
\\multicolumn{{5}}{{p{{0.92\\textwidth}}}}{{\\footnotesize Notes: Each cell reports the coefficient estimate with the clustered standard error below it. The baseline and foreign-placebo columns report single-regressor specifications; the current-plus-forward column reports the joint specification.}} \\\\
\\end{{tabular}}
\\end{{table}}
""".strip() + "\n"
    (OUT_DIR / "full_robustness_main_table.tex").write_text(tex)


def write_leave_one_country_out_table() -> None:
    df = pd.read_csv(OUT_DIR / "leave_one_country_out_coefficients.csv", keep_default_na=False)
    counts_path = OUT_DIR / "leave_one_country_out_sample_counts.csv"
    counts = pd.read_csv(counts_path, keep_default_na=False) if counts_path.exists() else pd.DataFrame()

    def pair(sample: str) -> tuple[pd.Series, pd.Series]:
        levels = df[(df["sample"] == sample) & (df["spec"] == "levels")].iloc[0]
        fd = df[(df["sample"] == sample) & (df["spec"] == "fd")].iloc[0]
        return levels, fd

    row_df = (
        df[["sample", "omitted_country"]]
        .drop_duplicates()
        .merge(
            counts[["spec", "omitted_rows"]] if not counts.empty else pd.DataFrame(columns=["spec", "omitted_rows"]),
            left_on="sample",
            right_on="spec",
            how="left",
        )
    )
    row_df["sort_omitted"] = row_df["omitted_rows"].fillna(0).astype(float)
    row_df["sort_full"] = row_df["omitted_country"].ne("None").astype(int)
    row_df = row_df.sort_values(["sort_full", "sort_omitted", "omitted_country"], ascending=[True, False, True])

    body = []
    for _, row in row_df.iterrows():
        label = str(row["omitted_country"])
        if label == "None":
            label = "Full sample"
        sample = str(row["sample"])
        levels, fd = pair(sample)
        body.append(
            f"{label} & {coef_cell(float(levels['estimate']), float(levels['std_error']))} & {int(levels['nobs'])} & "
            f"{coef_cell(float(fd['estimate']), float(fd['std_error']))} & {int(fd['nobs'])} \\\\"
        )

    tex = """
\\begin{table}[htbp]
\\centering
\\caption{Leave-one-country-out baseline pass-through regressions}
\\label{tab:leave_one_country_out_report}
\\begin{tabular}{lcccc}
\\toprule
Omitted country & Levels coef. & Levels $N$ & FD coef. & FD $N$ \\\\
\\midrule
""" + "\n".join(body) + """
\\bottomrule
\\end{tabular}
\\vspace{0.25em}
\\begin{minipage}{0.96\\textwidth}
\\footnotesize \\textit{Notes:} Each row re-estimates the baseline contemporaneous exchange-rate pass-through regression after omitting observations whose primary source country is listed in the first column. The full-sample row imposes no country omission. The levels specification includes make-model and year fixed effects and vehicle controls. The first-difference specification uses consecutive-year differences and year fixed effects. Coefficients are reported with make-model clustered standard errors in parentheses.
\\end{minipage}
\\end{table}
"""
    (OUT_DIR / "leave_one_country_out_report_table.tex").write_text(tex)


def main() -> None:
    write_main_identification_table()
    write_leave_one_country_out_table()
    print("Saved:")
    print(" -", OUT_DIR / "full_robustness_main_table.tex")
    print(" -", OUT_DIR / "leave_one_country_out_report_table.tex")


if __name__ == "__main__":
    main()
