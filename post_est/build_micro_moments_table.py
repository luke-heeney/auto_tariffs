from __future__ import annotations

import json
import pickle
import re
from pathlib import Path


def fmt_num(x: float) -> str:
    return f"{x:.3f}"


def fmt_diff(x: float) -> str:
    r = round(x, 3)
    if r == 0:
        return "$0.000$"
    sign = "+" if r > 0 else "-"
    return f"${sign}{abs(r):.3f}$"


def parse_micro_rows(formatted_moments: str) -> list[dict[str, float | str]]:
    pat = re.compile(
        r"\s*([+-]\d\.\d+E[+-]\d+)\s+([+-]\d\.\d+E[+-]\d+)\s+([+-]\d\.\d+E[+-]\d+)\s+(.*?)\s{2,}"
    )
    rows: list[dict[str, float | str]] = []
    for line in formatted_moments.splitlines():
        m = pat.match(line)
        if not m:
            continue
        rows.append(
            {
                "observed": float(m.group(1)),
                "estimated": float(m.group(2)),
                "difference": float(m.group(3)),
                "moment": m.group(4).strip(),
            }
        )
    return rows


def find_estimate(rows: list[dict[str, float | str]], pattern: str) -> float:
    hits = [r for r in rows if re.search(pattern, str(r["moment"]))]
    if len(hits) != 1:
        raise ValueError(f"Expected one micro-moment match for pattern `{pattern}`, found {len(hits)}.")
    return float(hits[0]["estimated"])


def build_row(label: str, observed_fixed: float, estimated: float) -> str:
    diff = observed_fixed - estimated
    return f"{label} & {fmt_num(observed_fixed)} & {fmt_num(estimated)} & {fmt_diff(diff)} \\\\"


def main() -> None:
    base = Path(__file__).resolve().parent
    cfg = json.loads((base / "results_config.json").read_text())
    results_path = Path(cfg["results_file"])
    if not results_path.is_absolute():
        results_path = (base / results_path).resolve()

    with open(results_path, "rb") as f:
        results = pickle.load(f)

    rows = parse_micro_rows(results._formatted_moments)

    panel_a = [
        (
            r"Mean price (Q2) $-$ Mean price (Q1)",
            0.011,
            r"^Mean price\(Q2\) - Mean price\(Q1\)",
        ),
        (
            r"Mean price (Q3) $-$ Mean price (Q1)",
            0.017,
            r"^Mean price\(Q3\) - Mean price\(Q1\)",
        ),
        (
            r"Mean price (Q4) $-$ Mean price (Q1)",
            0.011,
            r"^Mean price\(Q4\) - Mean price\(Q1\)",
        ),
        (
            r"Mean price (Q5) $-$ Mean price (Q1)",
            0.063,
            r"^Mean price\(Q5\) - Mean price\(Q1\)",
        ),
        (
            r"$P(\text{purchase}\mid Q_2)/P(\text{purchase}\mid Q_1)$",
            2.022,
            r"^P\(purchase\|Q2\)/P\(purchase\|Q1\)",
        ),
        (
            r"$P(\text{purchase}\mid Q_3)/P(\text{purchase}\mid Q_1)$",
            2.804,
            r"^P\(purchase\|Q3\)/P\(purchase\|Q1\)",
        ),
        (
            r"$P(\text{purchase}\mid Q_4)/P(\text{purchase}\mid Q_1)$",
            3.470,
            r"^P\(purchase\|Q4\)/P\(purchase\|Q1\)",
        ),
        (
            r"$P(\text{purchase}\mid Q_5)/P(\text{purchase}\mid Q_1)$",
            5.841,
            r"^P\(purchase\|Q5\)/P\(purchase\|Q1\)",
        ),
    ]

    panel_b = [
        (
            r"$P(\text{second is van}\mid \text{first is van})$",
            0.720,
            r"^P\(second is van \| first is van\)",
        ),
        (
            r"$P(\text{second is truck}\mid \text{first is truck})$",
            0.872,
            r"^P\(second is truck \| first is truck\)",
        ),
        (
            r"$P(\text{second is SUV}\mid \text{first is SUV})$",
            0.690,
            r"^P\(second is SUV \| first is SUV\)",
        ),
        (
            r"$P(\text{second is luxury}\mid \text{first is luxury})$",
            0.550,
            r"^P\(second is luxury \| first is luxury\)",
        ),
        (
            r"$\mathrm{corr}(\log \text{mpg}_{\text{first}},\log \text{mpg}_{\text{second}})$",
            0.611,
            r"^corr\(log_mpg_std_first, log_mpg_std_second\)",
        ),
        (
            r"$\mathrm{corr}(\log \text{hp}_{\text{first}},\log \text{hp}_{\text{second}})$",
            0.674,
            r"^corr\(log_hp_std_first, log_hp_std_second\)",
        ),
        (
            r"$P(\text{second is Euro-brand}\mid \text{first is Euro-brand})$",
            0.413,
            r"^P\(second is Euro-brand \| first is Euro-brand\)",
        ),
        (
            r"$P(\text{second is EV}\mid \text{first is EV})$",
            0.520,
            r"^P\(second is EV \| first is EV\)",
        ),
        (
            r"$P(\text{second is EV, same class}\mid \text{first is EV})$",
            0.330,
            r"^P\(second is EV in same class \| first is EV\)",
        ),
    ]

    panel_c = [
        (
            r"$P(\text{EV}\mid \text{purchase}, \text{Mountain}, 2021)$",
            0.032,
            r"^P\(EV \| purchase, div=Mountain, year=2021\)",
        ),
        (
            r"$P(\text{EV}\mid \text{purchase}, \text{North Central}, 2021)$",
            0.015,
            r"^P\(EV \| purchase, div=North Central, year=2021\)",
        ),
        (
            r"$P(\text{EV}\mid \text{purchase}, \text{North East}, 2021)$",
            0.025,
            r"^P\(EV \| purchase, div=North East, year=2021\)",
        ),
        (
            r"$P(\text{EV}\mid \text{purchase}, \text{Pacific}, 2021)$",
            0.083,
            r"^P\(EV \| purchase, div=Pacific, year=2021\)",
        ),
        (
            r"$P(\text{EV}\mid \text{purchase}, \text{South Atlantic}, 2021)$",
            0.024,
            r"^P\(EV \| purchase, div=South Atlantic, year=2021\)",
        ),
        (
            r"$P(\text{EV}\mid \text{purchase}, \text{South Central}, 2021)$",
            0.014,
            r"^P\(EV \| purchase, div=South Central, year=2021\)",
        ),
        (
            r"$P(\text{Truck}\mid \text{purchase}, \text{Mountain}, 2021)$",
            0.212,
            r"^P\(type=Truck \| purchase, div=Mountain, year=2021\)",
        ),
        (
            r"$P(\text{Truck}\mid \text{purchase}, \text{North Central}, 2021)$",
            0.191,
            r"^P\(type=Truck \| purchase, div=North Central, year=2021\)",
        ),
        (
            r"$P(\text{Truck}\mid \text{purchase}, \text{North East}, 2021)$",
            0.122,
            r"^P\(type=Truck \| purchase, div=North East, year=2021\)",
        ),
        (
            r"$P(\text{Truck}\mid \text{purchase}, \text{Pacific}, 2021)$",
            0.134,
            r"^P\(type=Truck \| purchase, div=Pacific, year=2021\)",
        ),
        (
            r"$P(\text{Truck}\mid \text{purchase}, \text{South Atlantic}, 2021)$",
            0.148,
            r"^P\(type=Truck \| purchase, div=South Atlantic, year=2021\)",
        ),
        (
            r"$P(\text{Truck}\mid \text{purchase}, \text{South Central}, 2021)$",
            0.212,
            r"^P\(type=Truck \| purchase, div=South Central, year=2021\)",
        ),
        (
            r"$P(\text{SUV}\mid \text{purchase}, \text{Mountain}, 2021)$",
            0.439,
            r"^P\(type=SUV \| purchase, div=Mountain, year=2021\)",
        ),
        (
            r"$P(\text{SUV}\mid \text{purchase}, \text{North Central}, 2021)$",
            0.505,
            r"^P\(type=SUV \| purchase, div=North Central, year=2021\)",
        ),
        (
            r"$P(\text{SUV}\mid \text{purchase}, \text{North East}, 2021)$",
            0.513,
            r"^P\(type=SUV \| purchase, div=North East, year=2021\)",
        ),
        (
            r"$P(\text{SUV}\mid \text{purchase}, \text{Pacific}, 2021)$",
            0.421,
            r"^P\(type=SUV \| purchase, div=Pacific, year=2021\)",
        ),
        (
            r"$P(\text{SUV}\mid \text{purchase}, \text{South Atlantic}, 2021)$",
            0.456,
            r"^P\(type=SUV \| purchase, div=South Atlantic, year=2021\)",
        ),
        (
            r"$P(\text{SUV}\mid \text{purchase}, \text{South Central}, 2021)$",
            0.425,
            r"^P\(type=SUV \| purchase, div=South Central, year=2021\)",
        ),
    ]

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Empirical and estimated micro-moments}",
        r"\label{tab:micro_moments}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{6pt}",
        r"\renewcommand{\arraystretch}{1.1}",
        r"\begin{threeparttable}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Moment & Observed & Estimated & Difference \\",
        r"\midrule",
        r"\multicolumn{4}{l}{\textit{Panel A. Income–price moments}} \\",
    ]

    for label, obs, pattern in panel_a[:4]:
        lines.append(build_row(label, obs, find_estimate(rows, pattern)))
    for label, obs, pattern in panel_a[4:]:
        lines.append(build_row(label, obs, find_estimate(rows, pattern)))

    lines.extend(
        [
            r"\midrule",
            r"\multicolumn{4}{l}{\textit{Panel B. Second-choice and match-on-characteristics moments}} \\[0.15em]",
            r"\multicolumn{4}{l}{\emph{2015 second-choice moments (from \cite{grieco_evolution_2024})}} \\",
        ]
    )

    for label, obs, pattern in panel_b[:7]:
        lines.append(build_row(label, obs, find_estimate(rows, pattern)))

    lines.extend(
        [
            r"[0.25em]",
            r"\multicolumn{4}{l}{\emph{2022 EV second-choice moments (from \cite{allcott_effects_2024})}} \\",
        ]
    )

    for label, obs, pattern in panel_b[7:]:
        lines.append(build_row(label, obs, find_estimate(rows, pattern)))

    lines.extend(
        [
            r"\midrule",
            r"\multicolumn{4}{l}{\textit{Panel C. EV and body-type shares by division, 2021}} \\",
            r"\multicolumn{4}{l}{\emph{EV share among purchasers}} \\",
        ]
    )

    for label, obs, pattern in panel_c[:6]:
        lines.append(build_row(label, obs, find_estimate(rows, pattern)))

    lines.extend(
        [
            r"[0.25em]",
            r"\multicolumn{4}{l}{\emph{Truck share among purchasers}} \\",
        ]
    )
    for label, obs, pattern in panel_c[6:12]:
        lines.append(build_row(label, obs, find_estimate(rows, pattern)))

    lines.extend(
        [
            r"[0.25em]",
            r"\multicolumn{4}{l}{\emph{SUV share among purchasers}} \\",
        ]
    )
    for label, obs, pattern in panel_c[12:]:
        lines.append(build_row(label, obs, find_estimate(rows, pattern)))

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{tablenotes}[flushleft]",
            r"\footnotesize",
            r"\item \textit{Notes:} Panel B: the first seven moments (van, truck, SUV, luxury, and the two correlation",
            r"moments plus Euro-brand) are for 2015 and are taken from \cite{grieco_evolution_2024}. The last two",
            r"Panel B moments (EV and EV in the same class) are for 2022 and are taken from \cite{allcott_effects_2024}.",
            r"Panel C: entries show a subset of division-level moments for 2021; analogous EV, truck, and SUV moments",
            r"are included for 2022–2024 but omitted here for brevity.",
            r"\end{tablenotes}",
            r"\end{threeparttable}",
            r"\end{table}",
        ]
    )

    out_dir = base / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "micro_moments.tex"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
