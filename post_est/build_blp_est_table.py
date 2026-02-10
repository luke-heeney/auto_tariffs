from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np


def fmt(x: float) -> str:
    return f"{x:.3f}"


def main() -> None:
    base = Path(__file__).resolve().parent
    cfg_path = base / "results_config.json"
    cfg = json.loads(cfg_path.read_text())

    results_path = Path(cfg["results_file"])
    if not results_path.is_absolute():
        results_path = (cfg_path.parent / results_path).resolve()

    with open(results_path, "rb") as f:
        results = pickle.load(f)

    sigma = np.asarray(results.sigma, dtype=float)
    sigma_se = np.asarray(results.sigma_se, dtype=float)
    pi = np.asarray(results.pi, dtype=float)
    pi_se = np.asarray(results.pi_se, dtype=float)
    beta = np.asarray(results.beta, dtype=float).reshape(-1)
    beta_se = np.asarray(results.beta_se, dtype=float).reshape(-1)

    sigma_labels = list(results.sigma_labels)
    pi_labels = list(results.pi_labels)
    beta_labels = list(results.beta_labels)

    s_idx = {k: i for i, k in enumerate(sigma_labels)}
    p_idx = {k: i for i, k in enumerate(pi_labels)}
    b_idx = {k: i for i, k in enumerate(beta_labels)}

    def sigma_diag(label: str) -> str:
        i = s_idx[label]
        return f"{fmt(sigma[i, i])} ({fmt(sigma_se[i, i])})"

    def pi_entry(s_label: str, p_label: str) -> str:
        i = s_idx[s_label]
        j = p_idx[p_label]
        return f"{fmt(pi[i, j])} ({fmt(pi_se[i, j])})"

    def beta_entry(label: str) -> str:
        i = b_idx[label]
        return f"{fmt(beta[i])} ({fmt(beta_se[i])})"

    panel_a_rows = [
        (r"\(\log(\text{mpg}_{\text{ICE/Hyb}})\)", sigma_diag("ln_mpg_ice")),
        (r"\(\log(\text{hp})\)", sigma_diag("log_hp_std")),
        (r"Van", sigma_diag("van_d")),
        (r"Truck", sigma_diag("truck_d")),
        (r"SUV", sigma_diag("suv_d")),
        (r"EV", sigma_diag("ev")),
        (r"Euro brand", sigma_diag("euro_brand")),
        (r"Luxury brand", sigma_diag("luxury_brand")),
    ]

    panel_b_rows = [
        (r"Intercept \(\times \log(\text{income}_{10k})\)", pi_entry("1", "log_income_10k")),
        (r"Price\(-\)subsidy \(\times \log(\text{income}_{10k})\)", pi_entry("prices - subsidy", "log_income_10k")),
        (r"Truck \(\times \) North Central", pi_entry("truck_d", "div_2")),
        (r"Truck \(\times\) South Central", pi_entry("truck_d", "div_4")),
        (r"Truck \(\times\) Mountain", pi_entry("truck_d", "div_5")),
        (r"SUV \(\times\) North East", pi_entry("suv_d", "div_1")),
        (r"SUV \(\times\) North Central", pi_entry("suv_d", "div_2")),
        (r"EV \(\times\) North East", pi_entry("ev", "div_1")),
        (r"EV \(\times\) South Atlantic", pi_entry("ev", "div_3")),
        (r"EV \(\times\) Mountain", pi_entry("ev", "div_5")),
        (r"EV \(\times \) Pacific", pi_entry("ev", "div_6")),
    ]

    panel_c_left = [
        (r"Price\(-\)subsidy", beta_entry("prices - subsidy")),
        (r"\(\log(\text{size})\)", beta_entry("log_size_std")),
        (r"\(\log(\text{hp})\)", beta_entry("log_hp_std")),
        (r"\(\log(\text{mpg}_{\text{ICE/hyb}})\)", beta_entry("ln_mpg_icehyb")),
        (r"\(\log(\text{mpg}_{\text{EV}})\)", beta_entry("ln_mpg_ev")),
        (r"Hybrid", beta_entry("hybrid")),
        (r"EV", beta_entry("ev")),
        (r"Van", beta_entry("van_d")),
        (r"Truck", beta_entry("truck_d")),
        (r"SUV", beta_entry("suv_d")),
    ]

    panel_c_right = []
    for year in range(2016, 2025):
        lab = f"ev*market_ids[{year}]"
        panel_c_right.append((rf"EV \(\times {year}\)", beta_entry(lab)))

    nrows = max(len(panel_c_left), len(panel_c_right))
    panel_c_rows: list[tuple[str, str, str, str]] = []
    for i in range(nrows):
        l_var, l_val = ("", "")
        r_var, r_val = ("", "")
        if i < len(panel_c_left):
            l_var, l_val = panel_c_left[i]
        if i < len(panel_c_right):
            r_var, r_val = panel_c_right[i]
        panel_c_rows.append((l_var, l_val, r_var, r_val))

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{BLP demand estimates with income and regional heterogeneity}",
        r"\label{tab:blp_est}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{6pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\begin{threeparttable}",
        "",
        r"\begin{tabular}{@{}l r l r@{}}",
        r"\toprule",
        r"\multicolumn{2}{@{}l}{\textbf{Panel A. Std.\ devs of random coefficients \(\sigma\)}} &",
        r"\multicolumn{2}{l}{\textbf{Panel B. Demographic heterogeneity \(\pi\)}} \\",
        r" & Coef (s.e.) & & Coef (s.e.) \\",
        r"\midrule",
    ]

    for i in range(max(len(panel_a_rows), len(panel_b_rows))):
        a_l, a_r = ("", "")
        b_l, b_r = ("", "")
        if i < len(panel_a_rows):
            a_l, a_r = panel_a_rows[i]
        if i < len(panel_b_rows):
            b_l, b_r = panel_b_rows[i]
        lines.append(f"{a_l} & {a_r} & {b_l} & {b_r} \\\\")

    lines.extend(
        [
            r"\midrule",
            r"\multicolumn{4}{@{}l}{\textbf{Panel C. Mean tastes \(\beta\) and EV\(\times\)year interactions}} \\",
            r"\multicolumn{2}{@{}l}{Mean tastes \(\beta\)} & \multicolumn{2}{l}{EV\(\times\)year interactions} \\",
            r"Variable & Coef (s.e.) & Term & Coef (s.e.) \\",
            r"\midrule",
        ]
    )

    for l_var, l_val, r_var, r_val in panel_c_rows:
        lines.append(f"{l_var} & {l_val} & {r_var} & {r_val} \\\\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            "",
            r"\begin{tablenotes}[flushleft]",
            r"\footnotesize",
            r"\item \textit{Notes:} ",
            r"Standard errors are clustered by model (2{,}982 clusters). \(\log(\text{income}_{10k})\) is household",
            r"income in \$10,000 units. Prices and subsidies are in \$100,000 units. Year and firm fixed effects and additional year-by-SUV, year-by-Hybrid interactions are included but",
            r"omitted from the table for brevity.",
            r"\end{tablenotes}",
            "",
            r"\end{threeparttable}",
            r"\end{table}",
        ]
    )

    out_dir = base / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "blp_est.tex"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
