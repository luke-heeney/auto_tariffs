from __future__ import annotations

import hashlib
import json
import pickle
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator, PercentFormatter

from country_normalization import normalize_country_series

try:
    import plotly.colors
    import plotly.graph_objects as go
except Exception:  # pragma: no cover - optional runtime dependency
    go = None


CANONICAL_DATE = "April 23, 2026"
RESULTS_CONFIG_PATH = Path("post_est/results_config.json")
CANONICAL_TEX_NAME = "a_main_draft_0_current_file.tex"
CANONICAL_PDF_NAME = "Auto_Tariffs.pdf"
REPORTING_BASELINE_LABEL = "no tariff (no subsidy)"

FULL_SCENARIOS = [
    ("B0", "no tariff (no subsidy)"),
    ("C1", "vehicles-only tariff (no subsidy)"),
    ("C2", "parts and vehicles tariff (no subsidy)"),
    ("C3", "no tariff (with subsidy)"),
    ("C4", "vehicles-only tariff (with subsidy)"),
    ("C5", "parts and vehicles tariff (with subsidy)"),
]

MAIN_SCENARIOS = ["C1", "C2", "C3"]
SCENARIO_LABELS = dict(FULL_SCENARIOS)
SCENARIO_CODES = {label: code for code, label in FULL_SCENARIOS}

SUMMARY_ROWS = [
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

STATE_ABBR = {
    "Alabama": "AL",
    "Alaska": "AK",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "Florida": "FL",
    "Georgia": "GA",
    "Hawaii": "HI",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Pennsylvania": "PA",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Washington": "WA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY",
    "District of Columbia": "DC",
}

STATE_CENTROIDS = {
    "AL": (32.806671, -86.791130),
    "AK": (61.370716, -152.404419),
    "AZ": (33.729759, -111.431221),
    "AR": (34.969704, -92.373123),
    "CA": (36.116203, -119.681564),
    "CO": (39.059811, -105.311104),
    "CT": (41.597782, -72.755371),
    "DE": (39.318523, -75.507141),
    "DC": (38.907200, -77.036900),
    "FL": (27.766279, -81.686783),
    "GA": (33.040619, -83.643074),
    "HI": (21.094318, -157.498337),
    "ID": (44.240459, -114.478828),
    "IL": (40.349457, -88.986137),
    "IN": (39.849426, -86.258278),
    "IA": (42.011539, -93.210526),
    "KS": (38.526600, -96.726486),
    "KY": (37.668140, -84.670067),
    "LA": (31.169546, -91.867805),
    "ME": (44.693947, -69.381927),
    "MD": (39.063946, -76.802101),
    "MA": (42.230171, -71.530106),
    "MI": (43.326618, -84.536095),
    "MN": (45.694454, -93.900192),
    "MS": (32.741646, -89.678696),
    "MO": (38.456085, -92.288368),
    "MT": (46.921925, -110.454353),
    "NE": (41.125370, -98.268082),
    "NV": (38.313515, -117.055374),
    "NH": (43.452492, -71.563896),
    "NJ": (40.298904, -74.521011),
    "NM": (34.840515, -106.248482),
    "NY": (42.165726, -74.948051),
    "NC": (35.630066, -79.806419),
    "ND": (47.528912, -99.784012),
    "OH": (40.388783, -82.764915),
    "OK": (35.565342, -96.928917),
    "OR": (44.572021, -122.070938),
    "PA": (40.590752, -77.209755),
    "RI": (41.680893, -71.511780),
    "SC": (33.856892, -80.945007),
    "SD": (44.299782, -99.438828),
    "TN": (35.747845, -86.692345),
    "TX": (31.054487, -97.563461),
    "UT": (40.150032, -111.862434),
    "VT": (44.045876, -72.710686),
    "VA": (37.769337, -78.169968),
    "WA": (47.400902, -121.490494),
    "WV": (38.491226, -80.954453),
    "WI": (44.268543, -89.616508),
    "WY": (42.755966, -107.302490),
}

SUMMARY_BY_TYPE_VALUES = {
    "counts": {"Car": 4178, "Truck": 455, "SUV": 3015, "Van": 403},
    "rows": [
        ("Sales", "Mean", [15686, 57234, 22869, 19918]),
        ("Sales", "Std. dev.", [33280, 92476, 38741, 25606]),
        ("Sales", "Min", [15, 38, 18, 18]),
        ("Sales", "Max", [334818, 529238, 374263, 130780]),
        ("Price (2015 USD, $100,000)", "Mean", [0.36, 0.35, 0.43, 0.31]),
        ("Price (2015 USD, $100,000)", "Std. dev.", [0.17, 0.07, 0.16, 0.05]),
        ("Price (2015 USD, $100,000)", "Min", [0.13, 0.19, 0.18, 0.21]),
        ("Price (2015 USD, $100,000)", "Max", [1.00, 0.74, 1.00, 0.47]),
        ("Horsepower (100s)", "Mean", [2.46, 3.41, 2.91, 2.65]),
        ("Horsepower (100s)", "Std. dev.", [1.04, 0.97, 0.89, 0.64]),
        ("Horsepower (100s)", "Min", [0.70, 1.91, 1.38, 1.31]),
        ("Horsepower (100s)", "Max", [8.45, 8.35, 8.35, 4.01]),
        ("Footprint (square ft, 100s)", "Mean", [0.97, 1.31, 1.06, 1.27]),
        ("Footprint (square ft, 100s)", "Std. dev.", [0.12, 0.19, 0.12, 0.20]),
        ("Footprint (square ft, 100s)", "Min", [0.45, 1.02, 0.81, 0.86]),
        ("Footprint (square ft, 100s)", "Max", [1.27, 1.84, 1.40, 1.92]),
        ("Curb weight (pounds, 1000s)", "Mean", [3.54, 5.06, 4.43, 4.57]),
        ("Curb weight (pounds, 1000s)", "Std. dev.", [0.57, 0.87, 0.76, 0.64]),
        ("Curb weight (pounds, 1000s)", "Min", [1.81, 3.52, 3.02, 3.25]),
        ("Curb weight (pounds, 1000s)", "Max", [5.92, 9.10, 6.92, 5.99]),
        ("Imported (%)", "Mean", [0.75, 0.08, 0.55, 0.59]),
        ("Electric (%)", "Mean", [0.06, 0.04, 0.05, 0.00]),
    ],
}

CANONICAL_BBL = r"""
\begin{thebibliography}{34}
\providecommand{\natexlab}[1]{#1}
\providecommand{\url}[1]{\texttt{#1}}
\providecommand{\urlprefix}{URL }

\bibitem[Allcott et~al.(2024)Allcott, Kane, Maydanchik, Shapiro, and
Tintelnot]{allcott_effects_2024}
Allcott, Hunt, Reigner Kane, Maximilian S. Maydanchik, Joseph S. Shapiro, and
Felix Tintelnot. 2024. ``The Effects of ``Buy American'': Electric Vehicles and
the Inflation Reduction Act.'' Working Paper 33032, National Bureau of
Economic Research. \url{https://www.nber.org/papers/w33032}.

\bibitem[Amiti et~al.(2019)Amiti, Redding, and Weinstein]{amiti2019tariff}
Amiti, Mary, Stephen J. Redding, and David E. Weinstein. 2019. ``The impact of
the 2018 tariffs on prices and welfare.'' \emph{Journal of Economic
Perspectives} 33 (4):187--210.

\bibitem[Amiti et~al.(2014)Amiti, Itskhoki, and Konings]{amiti_importers_2014}
Amiti, Mary, Oleg Itskhoki, and Jozef Konings. 2014. ``Importers, Exporters,
and Exchange Rate Disconnect.'' \emph{American Economic Review} 104
(7):1942--1978. \url{https://www.aeaweb.org/articles?id=10.1257/aer.104.7.1942}.

\bibitem[Antr\`as and Helpman(2004)]{antras2004global}
Antr\`as, Pol and Elhanan Helpman. 2004. ``Global sourcing.'' \emph{Journal of
Political Economy} 112 (3):552--580.

\bibitem[Bagwell and Staiger(1999)]{bagwell1999economic}
Bagwell, Kyle and Robert W. Staiger. 1999. ``An economic theory of GATT.''
\emph{American Economic Review} 89 (1):215--248.

\bibitem[Barrot and Sauvagnat(2016)]{barrot2016input}
Barrot, Jean-No\"el and Julien Sauvagnat. 2016. ``Input specificity and the
propagation of idiosyncratic shocks in production networks.'' \emph{The
Quarterly Journal of Economics} 131 (3):1543--1592.

\bibitem[Berry et~al.(1995)Berry, Levinsohn, and Pakes]{berry_automobile_1995}
Berry, Steven, James Levinsohn, and Ariel Pakes. 1995. ``Automobile Prices in
Market Equilibrium.'' \emph{Econometrica} 63 (4):841--890.
\url{https://www.jstor.org/stable/2171802}. Publisher: [Wiley, Econometric
Society].

\bibitem[Berry et~al.(2004)Berry, Levinsohn, and Pakes]{berry_differentiated_2004}
---. 2004. ``Differentiated Products Demand Systems from a Combination of Micro
and Macro Data: The New Car Market.'' \emph{Journal of Political Economy}
112 (1):68--105. \url{https://www.jstor.org/stable/10.1086/379939}. Publisher:
The University of Chicago Press.

\bibitem[Bureau of Labor Statistics(2025)]{bureau_of_labor_statistics_public_2025}
Bureau of Labor Statistics. 2025. ``Public Use Microdata (PUMD).''
\url{https://www.bls.gov/cex/pumd.htm}.

\bibitem[Conlon and Gortmaker(2020)]{conlon_best_2020}
Conlon, Christopher and Jeff Gortmaker. 2020. ``Best practices for
differentiated products demand estimation with PyBLP.'' \emph{The RAND Journal
of Economics} 51 (4):1108--1161.
\url{https://onlinelibrary.wiley.com/doi/abs/10.1111/1756-2171.12352}.

\bibitem[Conlon and Gortmaker(2025)]{conlon_incorporating_2025}
---. 2025. ``Incorporating Micro Data into Differentiated Products Demand
Estimation with PyBLP.'' \emph{Journal of Econometrics}:105926.
\url{https://www.sciencedirect.com/science/article/pii/S030440762400277X}.

\bibitem[Conlon and Mortimer(2021)]{conlon_empirical_2021}
Conlon, Christopher and Julie Holland Mortimer. 2021. ``Empirical Properties of
Diversion Ratios.'' \emph{The RAND Journal of Economics}.

\bibitem[Co\c{s}ar et~al.(2018)Co\c{s}ar, Grieco, Li, and
Tintelnot]{cosar_what_2018}
Co\c{s}ar, A. Kerem, Paul L. E. Grieco, Shengyu Li, and Felix Tintelnot. 2018.
``What drives home market advantage?'' \emph{Journal of International
Economics} 110:135--150.
\url{https://www.sciencedirect.com/science/article/pii/S0022199617301356}.

\bibitem[Department of Energy(2025)]{department_of_energy_tax_2025}
Department of Energy. 2025. ``Tax Incentive Data Services.''
\url{https://www.fueleconomy.gov/feg/ws/tax-data-services.shtml}.

\bibitem[Duarte et~al.(2025)Duarte, Magnolfi, Quint, Sullivan, and
S{\o}lvsten]{duarte_conduct_2025}
Duarte, Marco, Lorenzo Magnolfi, Daniel Quint, Christopher Sullivan, and Mikkel
S{\o}lvsten. 2025. ``Conduct and Scale Economies: Evaluating Tariffs in the US
Automobile Market.''

\bibitem[Burstein and Gopinath(2014)]{burstein_gopinath_2014}
Burstein, Ariel and Gita Gopinath. 2014. ``International Prices and Exchange
Rates.'' In \emph{Handbook of International Economics}, vol. 4, edited by Gita
Gopinath, Elhanan Helpman, and Kenneth Rogoff. Elsevier, 391--451.
\url{https://www.nber.org/papers/w18829}.

\bibitem[Froot and Klemperer(1989)]{froot_klemperer_1989}
Froot, Kenneth A. and Paul D. Klemperer. 1989. ``Exchange Rate Pass-Through
When Market Share Matters.'' \emph{American Economic Review} 79 (4):637--654.

\bibitem[Goldberg and Verboven(2001)]{goldberg_verboven_2001}
Goldberg, Pinelopi K. and Frank Verboven. 2001. ``The Evolution of Price
Dispersion in the European Car Market.'' \emph{Review of Economic Studies} 68
(4):811--848.

\bibitem[Goldberg and Knetter(1996)]{goldberg1996goods}
Goldberg, Pinelopi K. and Michael M. Knetter. 1996. ``Goods prices and exchange
rates: What have we learned?''

\bibitem[Gopinath et~al.(2010)Gopinath, Itskhoki, and Rigobon]{gopinath_currency_2010}
Gopinath, Gita, Oleg Itskhoki, and Roberto Rigobon. 2010. ``Currency Choice
and Exchange Rate Pass-Through.'' \emph{American Economic Review} 100
(1):304--336. \url{https://www.aeaweb.org/articles?id=10.1257/aer.100.1.304}.

\bibitem[Grieco et~al.(2024)Grieco, Murry, and Yurukoglu]{grieco_evolution_2024}
Grieco, Paul L. E., Charles Murry, and Ali Yurukoglu. 2024. ``The Evolution of
Market Power in the U.S. Automobile Industry.'' \emph{The Quarterly Journal of
Economics} 139 (2):1201--1253.
\url{https://doi.org/10.1093/qje/qjad047}.

\bibitem[Halton(1960)]{halton_efficiency_1960}
Halton, J. H. 1960. ``On the efficiency of certain quasi-random sequences of
points in evaluating multi-dimensional integrals.'' \emph{Numerische
Mathematik} 2 (1):84--90.
\url{http://link.springer.com/10.1007/BF01386213}.

\bibitem[Helper and Henderson(2014)]{helper_management_2014}
Helper, Susan and Rebecca Henderson. 2014. ``Management Practices, Relational
Contracts, and the Decline of General Motors.'' \emph{Journal of Economic
Perspectives} 28 (1):49--72.
\url{https://pubs.aeaweb.org/doi/10.1257/jep.28.1.49}.

\bibitem[Internal Revenue Service(2025)]{internal_revenue_service_clean_2025}
Internal Revenue Service. 2025. ``Clean vehicle tax credits | Internal Revenue
Service.'' \url{https://www.irs.gov/clean-vehicle-tax-credits}.

\bibitem[International Monetary Fund(2025)]{international_monetary_fund_international_2025}
International Monetary Fund. 2025. ``International Financial Statistics
database.'' \url{https://data.worldbank.org}.

\bibitem[Menk et~al.(2012)Menk, Chen, and Cregger]{menk_methodology_2012}
Menk, Debbie, Yen Chen, and Joshua Cregger. 2012. ``Methodology for Creating a
Matrix to Assess the Domestic Content of a Vehicle by Make and Model.'' Center
for Automotive Research.

\bibitem[Morrow and Skerlos(2011)]{morrow_fixed-point_2011}
Morrow, W. Ross and Steven J. Skerlos. 2011. ``Fixed-Point Approaches to
Computing Bertrand-Nash Equilibrium Prices Under Mixed-Logit Demand.''
\emph{Operations Research} 59 (2):328--345.
\url{https://pubsonline.informs.org/doi/10.1287/opre.1100.0894}. Publisher:
INFORMS.

\bibitem[Nevo(2001)]{nevo_measuring_2001}
Nevo, Aviv. 2001. ``Measuring Market Power in the Ready-to-Eat Cereal
Industry.'' \emph{Econometrica} 69 (2):307--342.
\url{https://onlinelibrary.wiley.com/doi/abs/10.1111/1468-0262.00194}.

\bibitem[Petrin(2002)]{petrin_quantifying_2002}
Petrin, Amil. 2002. ``Quantifying the Benefits of New Products: The Case of the
Minivan.'' \emph{Journal of Political Economy} 110 (4):705--729.
\url{https://www.jstor.org/stable/10.1086/340779}. Publisher: The University
of Chicago Press.

\bibitem[Ruggles et~al.(2025)Ruggles, Flood, Sobek, Backman, Cooper, Rivera
Drew, Richards, Rogers, Schroeder, and Williams]{ruggles_ipums_2025}
Ruggles, Steven, Sarah Flood, Matthew Sobek, Daniel Backman, Grace Cooper,
Julia A. Rivera Drew, Stephanie Richards, Renae Rogers, Jonathan Schroeder, and
Kari C. W. Williams. 2025. ``IPUMS USA.''

\bibitem[Rogoff(1996)]{rogoff_purchasing_1996}
Rogoff, Kenneth. 1996. ``The Purchasing Power Parity Puzzle.'' \emph{Journal
of Economic Literature} 34 (2):647--668.

\bibitem[Sabal(2025)]{sabal_product_2025}
Sabal, Alejandro. 2025. \emph{Product Entry in the Global Automobile Industry}.
Ph.D. thesis, Princeton University.

\bibitem[United States Congress(2025)]{united_states_congress_section_2025}
United States Congress. 2025. ``Section 232 Automotive Tariffs: Issues for
Congress.'' \url{https://www.congress.gov/crs-product/IN12545}.

\bibitem[WardsAuto(2025)]{wardsauto_most_2025}
WardsAuto. 2025. ``Most Automakers Anticipate Bright 2025 | WardsAuto.''
\url{https://www.wardsauto.com/news/most-automakers-anticipate-bright-2025/798797/}.

\end{thebibliography}
""".lstrip()


@dataclass(frozen=True)
class PaperPaths:
    repo_root: Path
    paper_dir: Path
    generated_dir: Path
    tables_dir: Path
    graphs_dir: Path
    build_dir: Path
    canonical_tex: Path
    canonical_pdf: Path
    results_config: Path


@dataclass(frozen=True)
class BundlePaths:
    source_bundle: Path
    rebased_bundle: Path
    graph_values_csv: Path


def discover_paths(repo_root: Path | None = None) -> PaperPaths:
    root = repo_root.resolve() if repo_root is not None else Path(__file__).resolve().parent.parent
    paper_dir = root / "paper"
    return PaperPaths(
        repo_root=root,
        paper_dir=paper_dir,
        generated_dir=paper_dir / "generated",
        tables_dir=paper_dir / "generated" / "tables",
        graphs_dir=paper_dir / "generated" / "graphs",
        build_dir=paper_dir / "build",
        canonical_tex=paper_dir / CANONICAL_TEX_NAME,
        canonical_pdf=paper_dir / CANONICAL_PDF_NAME,
        results_config=root / RESULTS_CONFIG_PATH,
    )


def ensure_generated_dirs(paths: PaperPaths) -> None:
    paths.generated_dir.mkdir(parents=True, exist_ok=True)
    paths.tables_dir.mkdir(parents=True, exist_ok=True)
    paths.graphs_dir.mkdir(parents=True, exist_ok=True)
    paths.build_dir.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_results_file(paths: PaperPaths) -> Path:
    cfg = json.loads(paths.results_config.read_text())
    results_path = Path(cfg["results_file"])
    if not results_path.is_absolute():
        results_path = (paths.results_config.parent / results_path).resolve()
    return results_path


def load_results(paths: PaperPaths):
    with _load_results_file(paths).open("rb") as fh:
        return pickle.load(fh)


def locate_rebased_bundle(paths: PaperPaths) -> BundlePaths:
    results_path = _load_results_file(paths).resolve()
    saved_outputs = paths.repo_root / "post_est" / "saved_outputs"
    candidates: list[tuple[float, Path, dict]] = []
    for bundle_dir in saved_outputs.iterdir():
        if not bundle_dir.is_dir():
            continue
        meta_path = bundle_dir / "metadata.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        if meta.get("reporting_baseline_label") != REPORTING_BASELINE_LABEL:
            continue
        meta_results = Path(str(meta.get("results_file", ""))).resolve()
        if meta_results != results_path:
            continue
        if bundle_dir.name.endswith("_rebased_b0_rebased_b0"):
            continue
        source_dir = Path(str(meta.get("source_saved_output_dir", "")))
        if not source_dir.is_absolute():
            source_dir = (paths.repo_root / source_dir).resolve()
        if not source_dir.exists():
            continue
        graph_values_csv = bundle_dir / "profit_changes_graph_values.csv"
        candidates.append((bundle_dir.stat().st_mtime, bundle_dir, {"source": source_dir, "graph_values": graph_values_csv}))
    if not candidates:
        raise FileNotFoundError("Could not find a rebased counterfactual output bundle for the canonical paper.")
    _, rebased_bundle, extra = max(candidates, key=lambda item: item[0])
    return BundlePaths(
        source_bundle=extra["source"],
        rebased_bundle=rebased_bundle,
        graph_values_csv=extra["graph_values"],
    )


def copy_generated_table(src: Path, dst: Path) -> None:
    dst.write_text(src.read_text())


def copy_generated_figure(src: Path, dst: Path) -> None:
    dst.write_bytes(src.read_bytes())


def copy_counterfactual_figures(bundle: BundlePaths, paths: PaperPaths) -> list[str]:
    figure_names = [
        "origin_metrics_vehicles_only_tariff__no_subsidy.png",
        "origin_metrics_parts_and_vehicles_tariff__no_subsidy.png",
        "profit_changes_vehicles_only_tariff__no_subsidy.png",
        "profit_changes_parts_and_vehicles_tariff__no_subsidy.png",
        "profit_changes_no_tariff__with_subsidy.png",
        "cs_map_no_tariff__with_subsidy.png",
        "cs_map_parts_and_vehicles_tariff__with_subsidy.png",
        "assembly_map_vehicles_only_tariff__no_subsidy.png",
        "assembly_map_parts_and_vehicles_tariff__no_subsidy.png",
        "assembly_map_no_tariff__no_subsidy.png",
        "profit_change_vs_import_share_parts_and_vehicles_tariff__with_subsidy.png",
    ]
    figure_dir = bundle.rebased_bundle / "figures"
    copied: list[str] = []
    for name in figure_names:
        src = figure_dir / name
        if not src.exists():
            raise FileNotFoundError(f"Missing expected counterfactual figure: {src}")
        copy_generated_figure(src, paths.graphs_dir / name)
        copied.append(name)
    return copied


def _results_market_arrays(results, market_id: int = 2024) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    products = getattr(results, "products", None) or results.problem.products
    market_ids = np.asarray(products.market_ids).reshape(-1)
    shares = np.asarray(products.shares, dtype=float).reshape(-1)
    mask = pd.to_numeric(market_ids, errors="coerce").reshape(-1) == market_id
    elasticities = results.compute_elasticities(market_id=market_id)
    own_elasticities = np.diag(elasticities)
    markups = results.compute_markups(market_id=market_id).reshape(-1)
    return own_elasticities, markups, shares[mask]


def build_price_coef_figure(paths: PaperPaths, out_name: str = "price_coef.png") -> None:
    results = load_results(paths)
    demographics = np.asarray(results.problem.agents.demographics, dtype=float)
    beta = np.asarray(results.beta, dtype=float).reshape(-1)
    pi = np.asarray(results.pi, dtype=float)

    alpha_i = float(beta[0]) + demographics @ pi[1, :]
    income = np.exp(demographics[:, 0])

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(alpha_i, bins=60, density=True, alpha=0.75, color="#4e79a7")
    axes[0].axvline(0.0, linestyle="--", linewidth=1.5, color="#2f2f2f")
    axes[0].set_title("Distribution of $\\alpha_i$")
    axes[0].set_xlabel("$\\alpha_i$")
    axes[0].set_ylabel("Density")

    axes[1].scatter(income, alpha_i, s=8, alpha=0.25, edgecolor="none", color="#e15759")
    axes[1].axhline(0.0, linestyle="--", linewidth=1.0, color="#2f2f2f")
    axes[1].set_title("$\\alpha_i$ vs. income")
    axes[1].set_xlabel("Income (10k)")
    axes[1].set_xlim(0, 50)
    axes[1].set_ylabel("$\\alpha_i$")

    fig.tight_layout()
    fig.savefig(paths.graphs_dir / out_name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_markup_distribution_figure(paths: PaperPaths, out_name: str = "markups_dist.png") -> None:
    results = load_results(paths)
    elas_2024, markups_2024, _ = _results_market_arrays(results, market_id=2024)

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(elas_2024[np.isfinite(elas_2024)], bins=60, density=True, alpha=0.75, color="#4e79a7")
    axes[0].set_title("Own-price elasticities (2024)")
    axes[0].set_xlabel("Own-price elasticity")
    axes[0].set_ylabel("Density")

    axes[1].hist(markups_2024[np.isfinite(markups_2024)], bins=60, density=True, alpha=0.75, color="#59a14f")
    axes[1].set_title("Markups (2024)")
    axes[1].set_xlabel("Markup")
    axes[1].set_ylabel("Density")

    fig.tight_layout()
    fig.savefig(paths.graphs_dir / out_name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_vehicle_type_trends_figure(paths: PaperPaths, out_names: Iterable[str]) -> None:
    df = pd.read_csv(paths.repo_root / "post_est" / "data" / "raw" / "product_data_45W.csv")
    df["market_year"] = pd.to_numeric(df["market_year"], errors="coerce")
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
    df["ev"] = pd.to_numeric(df["ev"], errors="coerce").fillna(0)
    df["hybrid"] = pd.to_numeric(df["hybrid"], errors="coerce").fillna(0)
    df = df.dropna(subset=["market_year", "sales"])
    df = df[df["sales"] > 0].copy()

    vt = df["vehicle_type"].astype(str).str.strip().str.lower()
    is_suv = vt.str.contains(r"\bsuv\b") | vt.str.contains("sport utility")
    is_truck = vt.str.contains(r"\btruck\b") | vt.str.contains("pickup")
    is_ev = df["ev"].astype(int) == 1
    is_hybrid = df["hybrid"].astype(int) == 1

    grouped = df.groupby("market_year", as_index=False).agg(
        total_sales=("sales", "sum"),
        ev_sales=("sales", lambda s: s[is_ev.loc[s.index]].sum()),
        hybrid_sales=("sales", lambda s: s[is_hybrid.loc[s.index]].sum()),
        suv_sales=("sales", lambda s: s[is_suv.loc[s.index]].sum()),
        truck_sales=("sales", lambda s: s[is_truck.loc[s.index]].sum()),
    )
    for key in ("ev", "hybrid", "suv", "truck"):
        grouped[f"{key}_share"] = grouped[f"{key}_sales"] / grouped["total_sales"]
    grouped = grouped.sort_values("market_year")

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(8, 4.5))
    series = [
        ("EV", "ev_share", "-", "#1f1f1f", 1.9),
        ("Hybrid", "hybrid_share", "--", "#555555", 1.9),
        ("SUV", "suv_share", ":", "#7f7f7f", 2.2),
        ("Truck", "truck_share", (0, (5, 2, 1, 2)), "#aaaaaa", 1.9),
    ]
    for label, col, linestyle, color, linewidth in series:
        ax.plot(grouped["market_year"], grouped[col], label=label, color=color, linestyle=linestyle, linewidth=linewidth)
    ax.set_xlabel("Year")
    ax.set_ylabel("Share of sales")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    for out_name in out_names:
        fig.savefig(paths.graphs_dir / out_name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_domestic_share_figure(paths: PaperPaths, out_name: str = "domestic_share.png") -> None:
    product_data = pd.read_csv(paths.repo_root / "post_est" / "data" / "raw" / "blp_with_45W_subsidies_scale1p0.csv")
    pc_panel = pd.read_csv(paths.repo_root / "post_est" / "data" / "raw" / "pc_data_panel.csv")

    df = product_data[["product_ids", "market_year", "sales", "home_mkt", "firm_ids"]].merge(
        pc_panel[["product_ids", "pcUSCA_pct"]], on="product_ids", how="left"
    )
    df = df[df["home_mkt"] == 1].copy()
    for col in ("market_year", "sales", "pcUSCA_pct"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["firm_ids"] = df["firm_ids"].astype(str).str.strip()
    df = df.dropna(subset=["market_year", "sales", "pcUSCA_pct", "firm_ids"])
    df = df[df["sales"] > 0].copy()
    if df["pcUSCA_pct"].max() > 1.0:
        df["pcUSCA_pct"] = df["pcUSCA_pct"] / 100.0

    top5 = (
        df.groupby("firm_ids", as_index=False)["sales"]
        .sum()
        .sort_values("sales", ascending=False)
        .head(5)["firm_ids"]
        .tolist()
    )
    tesla_key = next((firm for firm in df["firm_ids"].unique() if firm.lower() == "tesla"), None)
    brands = top5.copy()
    if tesla_key is not None and tesla_key not in brands:
        brands.append(tesla_key)

    def weighted_average(group: pd.DataFrame) -> float:
        return float(np.average(group["pcUSCA_pct"], weights=group["sales"]))

    brand_series = (
        df[df["firm_ids"].isin(brands)]
        .groupby(["market_year", "firm_ids"])
        .apply(weighted_average)
        .rename("pcUSCA_sales_weighted")
        .reset_index()
    )
    wide = (
        brand_series.pivot(index="market_year", columns="firm_ids", values="pcUSCA_sales_weighted")
        .reindex(columns=brands)
        .sort_index()
    )

    plt.rcParams.update({"font.family": "serif", "font.size": 10})
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    line_styles = ["-", "--", ":", "-.", (0, (5, 2)), (0, (1, 1))]
    line_widths = [2.0] + [1.6] * max(len(brands) - 1, 0)
    for idx, brand in enumerate(wide.columns):
        series = wide[brand].dropna()
        ax.plot(
            series.index.values,
            series.values,
            linestyle=line_styles[idx % len(line_styles)],
            linewidth=line_widths[idx],
            label=str(brand),
            color="black",
        )
    ax.set_xlabel("Year")
    ax.set_ylabel("US/Canada parts share")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(0, 1)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.grid(True, which="major", linewidth=0.6, alpha=0.35)
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=True, fontsize=9)
    leg.get_frame().set_linewidth(0.8)
    fig.tight_layout()
    fig.savefig(paths.graphs_dir / out_name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_sales_weighted_sources_figure(paths: PaperPaths, out_name: str = "sales_weighted_sources_by_year.png") -> None:
    df = pd.read_csv(paths.repo_root / "post_est" / "data" / "raw" / "product_data_45W.csv")
    df["market_year"] = pd.to_numeric(df["market_year"], errors="coerce")
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
    df["plant_country"] = normalize_country_series(df["plant_country"])
    df = df.dropna(subset=["market_year", "sales", "plant_country"])
    df = df[df["sales"] > 0].copy()

    top_k = 8
    force_other = {"United Kingdom", "Italy"}

    year_country_sales = df.groupby(["market_year", "plant_country"], as_index=False)["sales"].sum()
    total_by_country = (
        year_country_sales.groupby("plant_country", as_index=False)["sales"]
        .sum()
        .sort_values("sales", ascending=False)
    )
    top_countries = [c for c in total_by_country["plant_country"].head(top_k).tolist() if c not in force_other]

    yc = year_country_sales.copy()
    yc["plant_country"] = np.where(yc["plant_country"].isin(force_other), "Other", yc["plant_country"])
    yc["plant_country"] = np.where(yc["plant_country"].isin(top_countries), yc["plant_country"], "Other")
    yc = yc.groupby(["market_year", "plant_country"], as_index=False)["sales"].sum()

    wide = yc.pivot(index="market_year", columns="plant_country", values="sales").fillna(0)
    ordered_cols = [country for country in top_countries if country in wide.columns]
    if "Other" in wide.columns:
        ordered_cols.append("Other")
    wide = wide.reindex(columns=ordered_cols)
    shares = wide.div(wide.sum(axis=1).replace(0, np.nan), axis=0).fillna(0).sort_index()

    fig, ax = plt.subplots(figsize=(9, 4.5))
    cols = list(shares.columns)
    non_other = [c for c in cols if c != "Other"]
    blues = plt.cm.Blues(np.linspace(0.35, 0.85, max(len(non_other), 1)))[::-1]
    color_map = {country: blues[idx] for idx, country in enumerate(non_other)}
    color_map["Other"] = "#9aa0a6"
    ax.stackplot(
        shares.index.values,
        [shares[c].values for c in shares.columns],
        labels=shares.columns,
        colors=[color_map.get(c, "#9aa0a6") for c in shares.columns],
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Share of sales")
    ax.set_ylim(0, 1)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(paths.graphs_dir / out_name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_foreign_parts_figure(paths: PaperPaths, out_name: str = "foreign_parts.png") -> None:
    df = pd.read_csv(paths.repo_root / "cost_side" / "cost_side_panel_all.csv")
    df = df.loc[df["plant_country"].astype(str).eq("United States")].copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["pcUSCA_pct"] = pd.to_numeric(df["pcUSCA_pct"], errors="coerce")
    df["pcOth1_pct1"] = pd.to_numeric(df["pcOth1_pct1"], errors="coerce")
    df["pcOth1_code1"] = df["pcOth1_code1"].astype(str).str.strip()
    df = df.dropna(subset=["year"])

    year_counts = df.groupby("year", dropna=False).size().rename("n")
    foreign_total = (1.0 - df["pcUSCA_pct"]).groupby(df["year"]).mean()
    obs_sum = (
        df.groupby(["year", "pcOth1_code1"])["pcOth1_pct1"]
        .sum()
        .rename("sum_pct")
        .reset_index()
    )
    obs_sum = obs_sum.merge(year_counts.reset_index(), on="year", how="left")
    obs_sum["share"] = obs_sum["sum_pct"] / obs_sum["n"]
    obs_pivot = obs_sum.pivot(index="year", columns="pcOth1_code1", values="share").fillna(0.0).sort_index()
    observed_total = obs_pivot.sum(axis=1)
    unobserved = (foreign_total.reindex(obs_pivot.index) - observed_total).clip(lower=0.0)
    stack_df = obs_pivot.copy()
    stack_df["Unobserved"] = unobserved

    observed_cols = [c for c in stack_df.columns if c != "Unobserved"]
    order = stack_df[observed_cols].mean(axis=0).sort_values(ascending=False).index.tolist()
    labels = order + ["Unobserved"]
    values = [stack_df[c].to_numpy(dtype=float) for c in labels]
    colors = [
        "#4e79a7",
        "#f28e2b",
        "#59a14f",
        "#e15759",
        "#edc948",
        "#76b7b2",
        "#b07aa1",
        "#9c755f",
        "#bab0ac",
    ][: len(order)] + ["#b0b0b0"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="0.9", linewidth=0.8)
    ax.stackplot(
        stack_df.index.to_numpy(dtype=int),
        values,
        labels=labels,
        colors=colors,
        alpha=0.9,
        linewidth=0.6,
        edgecolor="white",
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Average foreign parts share")
    ax.set_ylim(0, max(1.0, float(foreign_total.max() if len(foreign_total) else 1.0)))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=True, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    fig.tight_layout()
    fig.savefig(paths.graphs_dir / out_name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_division_map(paths: PaperPaths, out_name: str = "div_map.png") -> None:
    if go is None:
        raise RuntimeError("Plotly is required to build the division map.")

    from post_est.helpers.figure_export import save_plotly_figure

    source_df = pd.read_csv(paths.repo_root / "post_est" / "data" / "raw" / "agent_data_cf.csv")
    div_cols = [col for col in source_df.columns if col.lower().startswith("div") and col[3:].isdigit()]
    if not div_cols:
        raise ValueError("No division columns found in agent_data_cf.csv.")

    state_raw = source_df["state"].astype(str).str.strip()
    state_key = np.where(state_raw.str.len() == 2, state_raw.str.upper(), state_raw.str.title())
    state_abbr = np.where(pd.Series(state_key).str.len() == 2, state_key, pd.Series(state_key).map(STATE_ABBR))
    df = source_df.copy()
    df["state_abbr"] = state_abbr
    df = df.dropna(subset=["state_abbr"])
    div_means = df.groupby("state_abbr", dropna=False)[div_cols].mean()
    div_idx = div_means.to_numpy().argmax(axis=1) + 1
    div_map = pd.DataFrame({"state_abbr": div_means.index, "division_id": div_idx})

    div_labels = ["North East", "North Central", "South Atlantic", "South Central", "Mountain", "Pacific"]
    palette = plotly.colors.qualitative.Set2[: len(div_labels)]
    colorscale = []
    for idx, color in enumerate(palette):
        frac = idx / max(len(palette) - 1, 1)
        colorscale.append([frac, color])

    fig = go.Figure(
        go.Choropleth(
            locations=div_map["state_abbr"],
            z=div_map["division_id"],
            locationmode="USA-states",
            colorscale=colorscale,
            showscale=False,
            marker_line_color="white",
            marker_line_width=0.5,
        )
    )
    for label, color in zip(div_labels, palette):
        fig.add_trace(
            go.Scattergeo(
                lon=[None],
                lat=[None],
                mode="markers",
                marker={"size": 10, "color": color},
                name=label,
                showlegend=True,
            )
        )
    labels = [(abbr, *STATE_CENTROIDS[abbr]) for abbr in div_map["state_abbr"] if abbr in STATE_CENTROIDS]
    if labels:
        fig.add_trace(
            go.Scattergeo(
                lon=[item[2] for item in labels],
                lat=[item[1] for item in labels],
                text=[item[0] for item in labels],
                mode="text",
                textposition="bottom center",
                textfont={"size": 10, "color": "black"},
                showlegend=False,
            )
        )
    fig.update_layout(
        title="US regional divisions",
        geo_scope="usa",
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        width=1100,
        height=700,
        legend={
            "x": 0.86,
            "y": 0.98,
            "xanchor": "left",
            "yanchor": "top",
            "bgcolor": "rgba(255,255,255,0.8)",
            "bordercolor": "rgba(0,0,0,0.1)",
            "borderwidth": 1,
            "font": {"size": 11},
        },
    )
    fig.update_geos(projection_scale=1.05, showframe=False, showcountries=False, showcoastlines=False)
    out_base = (paths.graphs_dir / out_name).with_suffix("")
    save_plotly_figure(fig, out_base)
    png_path = paths.graphs_dir / out_name
    if png_path.exists():
        return

    # Plotly static export is flaky in this environment; fall back to a plain
    # matplotlib lon/lat layout so the paper still builds from a clean checkout.
    fallback = div_map.copy()
    fallback["label"] = fallback["state_abbr"]
    fallback["lat"] = fallback["state_abbr"].map(lambda x: STATE_CENTROIDS.get(x, (np.nan, np.nan))[0])
    fallback["lon"] = fallback["state_abbr"].map(lambda x: STATE_CENTROIDS.get(x, (np.nan, np.nan))[1])
    fallback = fallback.dropna(subset=["lat", "lon"])

    palette = ["#66c2a5", "#8da0cb", "#fc8d62", "#e78ac3", "#a6d854", "#ffd92f"]
    fig2, ax = plt.subplots(figsize=(11, 7))
    for division_id, group in fallback.groupby("division_id"):
        color = palette[int(division_id) - 1]
        ax.scatter(group["lon"], group["lat"], s=35, color=color, alpha=0.9, label=div_labels[int(division_id) - 1])
        for _, row in group.iterrows():
            ax.text(row["lon"], row["lat"], row["label"], fontsize=8, ha="center", va="center")
    ax.set_title("US regional divisions")
    ax.set_xlim(-170, -65)
    ax.set_ylim(18, 72)
    ax.axis("off")
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper left", frameon=True, fontsize=10)
    fig2.tight_layout()
    fig2.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig2)


def build_diversions_table(paths: PaperPaths, out_name: str = "diversions_top5.tex") -> None:
    results = load_results(paths)
    labels = np.asarray(results.problem.products.clustering_ids).reshape(-1)
    market_ids = np.asarray(results.problem.products.market_ids).reshape(-1)
    labels = labels[pd.to_numeric(market_ids, errors="coerce") == 2024]
    diversion_matrix = pd.DataFrame(
        results.compute_diversion_ratios(market_id=2024),
        index=labels,
        columns=labels,
    )

    panel_a_pref = ["2024_tesla_model3ev", "2024_toyota_rav4hybrid", "2024_ford_bronco"]
    panel_b_pref = ["2024_audi_a5", "2024_honda_civic", "2024_ford_f150"]

    product_name_overrides = {
        "mercedesbenz_c": "Mercedes-Benz C-Class",
        "jeep_grandcherokee": "Jeep Grand Cherokee",
        "toyota_grandhighlanderhybrid": "Toyota Grand Highlander Hybrid",
        "lexus_rxhybrid": "Lexus RX Hybrid",
        "bmw_i4ev": "BMW i4 EV",
        "bmw_i5ev": "BMW i5 EV",
    }
    token_map = {
        "outside_good": "Outside Good",
        "tesla": "Tesla",
        "toyota": "Toyota",
        "ford": "Ford",
        "honda": "Honda",
        "audi": "Audi",
        "bmw": "BMW",
        "gmc": "GMC",
        "ram": "Ram",
        "jeep": "Jeep",
        "kia": "Kia",
        "hyundai": "Hyundai",
        "nissan": "Nissan",
        "chevrolet": "Chevrolet",
        "lucidmotors": "Lucid Motors",
        "mercedesbenz": "Mercedes-Benz",
        "volvo": "Volvo",
        "subaru": "Subaru",
        "lexus": "Lexus",
        "model3ev": "Model 3 EV",
        "modelyev": "Model Y EV",
        "modelsev": "Model S EV",
        "rav4hybrid": "RAV4 Hybrid",
        "cclass": "C-Class",
        "3series": "3 Series",
        "4series": "4 Series",
        "5series": "5 Series",
        "f150": "F-150",
        "crvhybrid": "CR-V Hybrid",
    }

    def format_token(token: str) -> str:
        if token in token_map:
            return token_map[token]
        if token.endswith("ev") and len(token) > 2:
            return f"{token[:-2].title()} EV"
        if token.endswith("hybrid") and len(token) > 6:
            return f"{token[:-6].title()} Hybrid"
        return token.title()

    def display_name(product_id: str) -> str:
        if product_id == "outside_good":
            return "Outside Good"
        pid = product_id[5:] if product_id.startswith("2024_") else product_id
        if pid in product_name_overrides:
            return product_name_overrides[pid]
        return " ".join(format_token(token) for token in pid.split("_"))

    def pick_three(preferred: list[str], used: set[str]) -> list[str]:
        available = list(diversion_matrix.index)
        picked = [pid for pid in preferred if pid in diversion_matrix.index and pid not in used]
        if len(picked) < 3:
            picked.extend([pid for pid in available if pid not in used and pid not in picked][: 3 - len(picked)])
        return picked[:3]

    def diversion_rows(product_id: str, top_n: int = 5) -> list[tuple[str, float]]:
        row = diversion_matrix.loc[product_id].copy()
        outside = float(row[product_id])
        top = row.drop(index=product_id).sort_values(ascending=False).head(top_n)
        entries = [("Outside Good", outside)]
        entries.extend((display_name(pid), float(share)) for pid, share in top.items())
        return entries

    def panel_lines(product_ids: list[str]) -> list[str]:
        headers = [display_name(pid) for pid in product_ids]
        panel = [diversion_rows(pid, top_n=5) for pid in product_ids]
        lines = [
            r"\begin{tabular}{@{}p{0.27\textwidth}r@{\hspace{6pt}}|@{\hspace{6pt}}p{0.27\textwidth}r@{\hspace{6pt}}|@{\hspace{6pt}}p{0.27\textwidth}r@{}}",
            r"\toprule",
            rf"\multicolumn{{2}}{{c}}{{\textbf{{{headers[0]}}}}} & \multicolumn{{2}}{{c}}{{\textbf{{{headers[1]}}}}} & \multicolumn{{2}}{{c}}{{\textbf{{{headers[2]}}}}} \\",
            r"\cmidrule(lr){1-2}\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
            r"\textbf{Alternative} & \textbf{Share} & \textbf{Alternative} & \textbf{Share} & \textbf{Alternative} & \textbf{Share} \\",
            r"\midrule",
        ]
        for idx in range(6):
            a1, s1 = panel[0][idx]
            a2, s2 = panel[1][idx]
            a3, s3 = panel[2][idx]
            lines.append(f"{a1} & {100.0 * s1:.1f}\\% & {a2} & {100.0 * s2:.1f}\\% & {a3} & {100.0 * s3:.1f}\\% \\\\")
        lines.extend([r"\bottomrule", r"\end{tabular}"])
        return lines

    panel_a_ids = pick_three(panel_a_pref, set())
    panel_b_ids = pick_three(panel_b_pref, set(panel_a_ids))

    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.1}",
        "",
        r"% Panel A",
        *panel_lines(panel_a_ids),
        "",
        r"\vspace{0.6em}",
        "",
        r"% Panel B",
        *panel_lines(panel_b_ids),
        "",
        r"\caption{Outside-good diversion and top five diversion destinations (2024)}",
        r"\label{tab:diversions_top5}",
        r"\end{table}",
        "",
    ]
    (paths.tables_dir / out_name).write_text("\n".join(lines))


def _load_rebased_summary(bundle: BundlePaths) -> pd.DataFrame:
    summary_path = bundle.rebased_bundle / "summary_tbl_all.csv.gz"
    summary = pd.read_csv(summary_path).set_index("Unnamed: 0")
    summary = summary[[label for _, label in FULL_SCENARIOS]]
    return summary.reindex(SUMMARY_ROWS)


def _parse_cs_cell(value: object) -> tuple[float, float]:
    match = re.fullmatch(r"\s*([+-]?\d+(?:\.\d+)?)\s*\(([-+]?\d+(?:\.\d+)?)%\)\s*", str(value))
    if match is None:
        raise ValueError(f"Could not parse consumer-surplus cell: {value!r}")
    return float(match.group(1)), float(match.group(2))


def _format_cs(value: object) -> str:
    return str(value).replace("%", r"\%")


def _fmt_float(value: object, digits: int) -> str:
    return f"{float(value):.{digits}f}"


def build_counterfactual_tables(paths: PaperPaths, bundle: BundlePaths) -> None:
    summary = _load_rebased_summary(bundle)
    summary.columns = [SCENARIO_CODES[col] for col in summary.columns]
    summary = summary[[code for code, _ in FULL_SCENARIOS]]

    markup_levels = pd.to_numeric(summary.loc["Sales-weighted Markup (CF, %)"], errors="coerce")
    delta_markup = markup_levels - float(markup_levels["B0"])

    def cs_row(row_name: str, code: str) -> str:
        return _format_cs(summary.loc[row_name, code])

    def us_share(code: str) -> str:
        return _fmt_float(100.0 * float(summary.loc["US share of vehicles sold (CF)", code]), 1)

    full_lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Counterfactual Tariff and Subsidy Scenarios: 2024 Market Outcomes}",
        r"\label{tab:cf_summary_full}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.10}",
        r"\begin{adjustbox}{max width=\textwidth}",
        r"\begin{threeparttable}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r" & \multicolumn{3}{c}{Without Subsidy} & \multicolumn{3}{c}{With Subsidy} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r"Tariff Status & B0: No & C1: Vehicles-only & C2: Parts \& vehicle & C3: No & C4: Vehicles-only & C5: Parts \& vehicle \\",
        r"\midrule",
        rf"$\Delta$ Price (avg, \%) & {_fmt_float(summary.loc['Sales-weighted Δ Price (%)', 'B0'], 2)} & {_fmt_float(summary.loc['Sales-weighted Δ Price (%)', 'C1'], 2)} & {_fmt_float(summary.loc['Sales-weighted Δ Price (%)', 'C2'], 2)} & {_fmt_float(summary.loc['Sales-weighted Δ Price (%)', 'C3'], 2)} & {_fmt_float(summary.loc['Sales-weighted Δ Price (%)', 'C4'], 2)} & {_fmt_float(summary.loc['Sales-weighted Δ Price (%)', 'C5'], 2)} \\",
        rf"Markup (avg \%) & {_fmt_float(markup_levels['B0'], 1)} & {_fmt_float(markup_levels['C1'], 1)} & {_fmt_float(markup_levels['C2'], 1)} & {_fmt_float(markup_levels['C3'], 1)} & {_fmt_float(markup_levels['C4'], 1)} & {_fmt_float(markup_levels['C5'], 1)} \\",
        rf"$\Delta$ Markup (avg \%) & {_fmt_float(delta_markup['B0'], 1)} & {_fmt_float(delta_markup['C1'], 1)} & {_fmt_float(delta_markup['C2'], 1)} & {_fmt_float(delta_markup['C3'], 1)} & {_fmt_float(delta_markup['C4'], 1)} & {_fmt_float(delta_markup['C5'], 1)} \\",
        rf"$\Delta$ US Producer Surplus (b USD) & {_fmt_float(summary.loc['US Producer Surplus (Δ, billion USD)', 'B0'], 2)} & {_fmt_float(summary.loc['US Producer Surplus (Δ, billion USD)', 'C1'], 2)} & {_fmt_float(summary.loc['US Producer Surplus (Δ, billion USD)', 'C2'], 2)} & {_fmt_float(summary.loc['US Producer Surplus (Δ, billion USD)', 'C3'], 2)} & {_fmt_float(summary.loc['US Producer Surplus (Δ, billion USD)', 'C4'], 2)} & {_fmt_float(summary.loc['US Producer Surplus (Δ, billion USD)', 'C5'], 2)} \\",
        rf"CS $\Delta$ total (b USD) & {cs_row('CS Δ total (billion USD)', 'B0')} & {cs_row('CS Δ total (billion USD)', 'C1')} & {cs_row('CS Δ total (billion USD)', 'C2')} & {cs_row('CS Δ total (billion USD)', 'C3')} & {cs_row('CS Δ total (billion USD)', 'C4')} & {cs_row('CS Δ total (billion USD)', 'C5')} \\",
        rf"CS $\Delta$ Q1 (b USD) & {cs_row('CS Δ Q1 (billion USD)', 'B0')} & {cs_row('CS Δ Q1 (billion USD)', 'C1')} & {cs_row('CS Δ Q1 (billion USD)', 'C2')} & {cs_row('CS Δ Q1 (billion USD)', 'C3')} & {cs_row('CS Δ Q1 (billion USD)', 'C4')} & {cs_row('CS Δ Q1 (billion USD)', 'C5')} \\",
        rf"CS $\Delta$ Q2 (b USD) & {cs_row('CS Δ Q2 (billion USD)', 'B0')} & {cs_row('CS Δ Q2 (billion USD)', 'C1')} & {cs_row('CS Δ Q2 (billion USD)', 'C2')} & {cs_row('CS Δ Q2 (billion USD)', 'C3')} & {cs_row('CS Δ Q2 (billion USD)', 'C4')} & {cs_row('CS Δ Q2 (billion USD)', 'C5')} \\",
        rf"CS $\Delta$ Q3 (b USD) & {cs_row('CS Δ Q3 (billion USD)', 'B0')} & {cs_row('CS Δ Q3 (billion USD)', 'C1')} & {cs_row('CS Δ Q3 (billion USD)', 'C2')} & {cs_row('CS Δ Q3 (billion USD)', 'C3')} & {cs_row('CS Δ Q3 (billion USD)', 'C4')} & {cs_row('CS Δ Q3 (billion USD)', 'C5')} \\",
        rf"CS $\Delta$ Q4 (b USD) & {cs_row('CS Δ Q4 (billion USD)', 'B0')} & {cs_row('CS Δ Q4 (billion USD)', 'C1')} & {cs_row('CS Δ Q4 (billion USD)', 'C2')} & {cs_row('CS Δ Q4 (billion USD)', 'C3')} & {cs_row('CS Δ Q4 (billion USD)', 'C4')} & {cs_row('CS Δ Q4 (billion USD)', 'C5')} \\",
        rf"CS $\Delta$ Q5 (b USD) & {cs_row('CS Δ Q5 (billion USD)', 'B0')} & {cs_row('CS Δ Q5 (billion USD)', 'C1')} & {cs_row('CS Δ Q5 (billion USD)', 'C2')} & {cs_row('CS Δ Q5 (billion USD)', 'C3')} & {cs_row('CS Δ Q5 (billion USD)', 'C4')} & {cs_row('CS Δ Q5 (billion USD)', 'C5')} \\",
        rf"$\Delta$ vehicles sold (m) & {_fmt_float(summary.loc['Δ vehicles sold (millions)', 'B0'], 3)} & {_fmt_float(summary.loc['Δ vehicles sold (millions)', 'C1'], 3)} & {_fmt_float(summary.loc['Δ vehicles sold (millions)', 'C2'], 3)} & {_fmt_float(summary.loc['Δ vehicles sold (millions)', 'C3'], 3)} & {_fmt_float(summary.loc['Δ vehicles sold (millions)', 'C4'], 3)} & {_fmt_float(summary.loc['Δ vehicles sold (millions)', 'C5'], 3)} \\",
        rf"EV share (\% sales) & {_fmt_float(summary.loc['EV share of vehicles sold (CF, %)', 'B0'], 2)} & {_fmt_float(summary.loc['EV share of vehicles sold (CF, %)', 'C1'], 2)} & {_fmt_float(summary.loc['EV share of vehicles sold (CF, %)', 'C2'], 2)} & {_fmt_float(summary.loc['EV share of vehicles sold (CF, %)', 'C3'], 2)} & {_fmt_float(summary.loc['EV share of vehicles sold (CF, %)', 'C4'], 2)} & {_fmt_float(summary.loc['EV share of vehicles sold (CF, %)', 'C5'], 2)} \\",
        rf"US-assembled share (\% sales) & {us_share('B0')} & {us_share('C1')} & {us_share('C2')} & {us_share('C3')} & {us_share('C4')} & {us_share('C5')} \\",
        rf"$\Delta$ US assembled (m) & {_fmt_float(summary.loc['Δ US assembled (millions)', 'B0'], 3)} & {_fmt_float(summary.loc['Δ US assembled (millions)', 'C1'], 3)} & {_fmt_float(summary.loc['Δ US assembled (millions)', 'C2'], 3)} & {_fmt_float(summary.loc['Δ US assembled (millions)', 'C3'], 3)} & {_fmt_float(summary.loc['Δ US assembled (millions)', 'C4'], 3)} & {_fmt_float(summary.loc['Δ US assembled (millions)', 'C5'], 3)} \\",
        rf"Tariff revenue (b USD) & {_fmt_float(summary.loc['Tariff revenue (billion USD)', 'B0'], 3)} & {_fmt_float(summary.loc['Tariff revenue (billion USD)', 'C1'], 2)} & {_fmt_float(summary.loc['Tariff revenue (billion USD)', 'C2'], 2)} & {_fmt_float(summary.loc['Tariff revenue (billion USD)', 'C3'], 3)} & {_fmt_float(summary.loc['Tariff revenue (billion USD)', 'C4'], 2)} & {_fmt_float(summary.loc['Tariff revenue (billion USD)', 'C5'], 2)} \\",
        rf"EV subsidy cost (b USD) & {_fmt_float(summary.loc['EV subsidy spending (billion USD)', 'B0'], 3)} & {_fmt_float(summary.loc['EV subsidy spending (billion USD)', 'C1'], 3)} & {_fmt_float(summary.loc['EV subsidy spending (billion USD)', 'C2'], 3)} & {_fmt_float(summary.loc['EV subsidy spending (billion USD)', 'C3'], 2)} & {_fmt_float(summary.loc['EV subsidy spending (billion USD)', 'C4'], 2)} & {_fmt_float(summary.loc['EV subsidy spending (billion USD)', 'C5'], 2)} \\",
        rf"$\Delta$ Net US outcomes (b USD) & {_fmt_float(summary.loc['Net US impact (billion USD)', 'B0'], 2)} & {_fmt_float(summary.loc['Net US impact (billion USD)', 'C1'], 2)} & {_fmt_float(summary.loc['Net US impact (billion USD)', 'C2'], 2)} & {_fmt_float(summary.loc['Net US impact (billion USD)', 'C3'], 2)} & {_fmt_float(summary.loc['Net US impact (billion USD)', 'C4'], 2)} & {_fmt_float(summary.loc['Net US impact (billion USD)', 'C5'], 2)} \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}[flushleft]\footnotesize",
        r"\item \textit{Notes:} $\Delta$ entries report counterfactual outcomes relative to the `no tariff, EV subsidy repealed' baseline. Dollars are USD 2015. $\Delta$ Net US outcomes is the change in US producer and consumer surplus, plus tariff revenue, minus (plus) additional EV subsidy expenditure (savings) compared to baseline. US Producer Surplus counts profit changes for US-headquartered firms. Consumer surplus (CS) changes are in billion USD; parentheses report percentage changes. `EV subsidy reinstated' and `EV subsidy repealed' refer to whether the EV subsidy policy is in place in the counterfactual.",
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{adjustbox}",
        r"\end{table}",
        "",
    ]
    (paths.tables_dir / "cf_summary_table_full.tex").write_text("\n".join(full_lines))

    main_lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Counterfactual Tariff and Subsidy Scenarios: 2024 Market Outcomes}",
        r"\label{tab:cf_summary}",
        r"\small",
        r"\setlength{\tabcolsep}{6pt}",
        r"\renewcommand{\arraystretch}{1.10}",
        r"\begin{threeparttable}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r" & \multicolumn{2}{c}{Without Subsidy} & With Subsidy \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-4}",
        r" & C1: Vehicles-only & C2: Parts \& vehicle & C3: No tariff \\",
        r"\midrule",
        rf"$\Delta$ Price (avg, \%) & {_fmt_float(summary.loc['Sales-weighted Δ Price (%)', 'C1'], 2)} & {_fmt_float(summary.loc['Sales-weighted Δ Price (%)', 'C2'], 2)} & {_fmt_float(summary.loc['Sales-weighted Δ Price (%)', 'C3'], 2)} \\",
        rf"$\Delta$ Markup (avg \%) & {_fmt_float(delta_markup['C1'], 1)} & {_fmt_float(delta_markup['C2'], 1)} & {_fmt_float(delta_markup['C3'], 1)} \\",
        rf"$\Delta$ US Producer Surplus (b USD) & {_fmt_float(summary.loc['US Producer Surplus (Δ, billion USD)', 'C1'], 2)} & {_fmt_float(summary.loc['US Producer Surplus (Δ, billion USD)', 'C2'], 2)} & {_fmt_float(summary.loc['US Producer Surplus (Δ, billion USD)', 'C3'], 2)} \\",
        rf"CS $\Delta$ total (b USD) & {cs_row('CS Δ total (billion USD)', 'C1')} & {cs_row('CS Δ total (billion USD)', 'C2')} & {cs_row('CS Δ total (billion USD)', 'C3')} \\",
        rf"CS $\Delta$ Q1 (b USD) & {cs_row('CS Δ Q1 (billion USD)', 'C1')} & {cs_row('CS Δ Q1 (billion USD)', 'C2')} & {cs_row('CS Δ Q1 (billion USD)', 'C3')} \\",
        rf"CS $\Delta$ Q2 (b USD) & {cs_row('CS Δ Q2 (billion USD)', 'C1')} & {cs_row('CS Δ Q2 (billion USD)', 'C2')} & {cs_row('CS Δ Q2 (billion USD)', 'C3')} \\",
        rf"CS $\Delta$ Q3 (b USD) & {cs_row('CS Δ Q3 (billion USD)', 'C1')} & {cs_row('CS Δ Q3 (billion USD)', 'C2')} & {cs_row('CS Δ Q3 (billion USD)', 'C3')} \\",
        rf"CS $\Delta$ Q4 (b USD) & {cs_row('CS Δ Q4 (billion USD)', 'C1')} & {cs_row('CS Δ Q4 (billion USD)', 'C2')} & {cs_row('CS Δ Q4 (billion USD)', 'C3')} \\",
        rf"CS $\Delta$ Q5 (b USD) & {cs_row('CS Δ Q5 (billion USD)', 'C1')} & {cs_row('CS Δ Q5 (billion USD)', 'C2')} & {cs_row('CS Δ Q5 (billion USD)', 'C3')} \\",
        rf"$\Delta$ vehicles sold (m) & {_fmt_float(summary.loc['Δ vehicles sold (millions)', 'C1'], 3)} & {_fmt_float(summary.loc['Δ vehicles sold (millions)', 'C2'], 3)} & {_fmt_float(summary.loc['Δ vehicles sold (millions)', 'C3'], 3)} \\",
        rf"EV share (\% sales) & {_fmt_float(summary.loc['EV share of vehicles sold (CF, %)', 'C1'], 2)} & {_fmt_float(summary.loc['EV share of vehicles sold (CF, %)', 'C2'], 2)} & {_fmt_float(summary.loc['EV share of vehicles sold (CF, %)', 'C3'], 2)} \\",
        rf"US-assembled share (\% sales) & {us_share('C1')} & {us_share('C2')} & {us_share('C3')} \\",
        rf"$\Delta$ US assembled (m) & {_fmt_float(summary.loc['Δ US assembled (millions)', 'C1'], 3)} & {_fmt_float(summary.loc['Δ US assembled (millions)', 'C2'], 3)} & {_fmt_float(summary.loc['Δ US assembled (millions)', 'C3'], 3)} \\",
        rf"Tariff revenue (b USD) & {_fmt_float(summary.loc['Tariff revenue (billion USD)', 'C1'], 2)} & {_fmt_float(summary.loc['Tariff revenue (billion USD)', 'C2'], 2)} & {_fmt_float(summary.loc['Tariff revenue (billion USD)', 'C3'], 3)} \\",
        rf"EV subsidy cost (b USD) & {_fmt_float(summary.loc['EV subsidy spending (billion USD)', 'C1'], 3)} & {_fmt_float(summary.loc['EV subsidy spending (billion USD)', 'C2'], 3)} & {_fmt_float(summary.loc['EV subsidy spending (billion USD)', 'C3'], 2)} \\",
        rf"$\Delta$ Net US outcomes (b USD) & {_fmt_float(summary.loc['Net US impact (billion USD)', 'C1'], 2)} & {_fmt_float(summary.loc['Net US impact (billion USD)', 'C2'], 2)} & {_fmt_float(summary.loc['Net US impact (billion USD)', 'C3'], 2)} \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{threeparttable}",
        r"\end{table}",
        "",
    ]
    (paths.tables_dir / "cf_summary_table.tex").write_text("\n".join(main_lines))


def build_summary_by_type_table(paths: PaperPaths, out_name: str = "summary_by_type.tex") -> None:
    counts = SUMMARY_BY_TYPE_VALUES["counts"]
    rows = SUMMARY_BY_TYPE_VALUES["rows"]

    def format_value(value: object) -> str:
        if isinstance(value, int):
            return f"{value:,}"
        return f"{float(value):.2f}"

    def escape_tex(text: str) -> str:
        replacements = {
            "\\": r"\textbackslash{}",
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "_": r"\_",
            "{": r"\{",
            "}": r"\}",
        }
        escaped = text
        for src, dst in replacements.items():
            escaped = escaped.replace(src, dst)
        return escaped

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Summary Statistics by Vehicle Type}",
        r"\label{tab:summary_by_type}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{6pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\begin{adjustbox}{max width=\textwidth}",
        r"\begin{tabular}{llllll}",
        r"\toprule",
        rf" &  & Cars, $N = {counts['Car']}$ & Trucks, $N = {counts['Truck']}$ & SUVs, $N = {counts['SUV']}$ & Vans, $N = {counts['Van']}$ \\",
        r"\midrule",
    ]

    sections = []
    seen = []
    for section, _, _ in rows:
        if section not in seen:
            seen.append(section)
    for section in seen:
        sections.append((section, [row for row in rows if row[0] == section]))

    for section_idx, (section, section_rows) in enumerate(sections):
        for row_idx, (_, stat, values) in enumerate(section_rows):
            section_cell = rf"\multirow{{{len(section_rows)}}}{{*}}{{{escape_tex(section)}}}" if row_idx == 0 else " "
            values_str = " & ".join(format_value(value) for value in values)
            lines.append(rf"{section_cell} & {stat} & {values_str} \\")
        if section_idx < len(sections) - 1:
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}", ""])
    (paths.tables_dir / out_name).write_text("\n".join(lines))


def build_profit_scatter_figures(paths: PaperPaths, bundle: BundlePaths) -> None:
    if bundle.graph_values_csv.exists():
        graph_values = pd.read_csv(bundle.graph_values_csv)
    else:
        scenario_index = pd.read_csv(bundle.rebased_bundle / "scenario_index.csv")
        us_firms = {
            "ford",
            "chevrolet",
            "gmc",
            "buick",
            "cadillac",
            "chrysler",
            "ram",
            "jeep",
            "dodge",
            "tesla",
            "rivian",
            "lucid",
            "lincoln",
            "lucidmotors",
        }
        rows: list[pd.DataFrame] = []
        for _, scenario in scenario_index.iterrows():
            slug = str(scenario["label_slug"])
            firm_table = pd.read_csv(bundle.rebased_bundle / f"{slug}__firm_table.csv.gz").copy()
            firm_table["scenario_key"] = str(scenario["scenario_key"])
            firm_table["scenario_label"] = str(scenario["scenario_label"])
            firm_table["label_slug"] = slug
            firm_table["firm_ids"] = firm_table["firm_ids"].astype(str)
            firm_table["plotted_firm_label"] = firm_table["firm_ids"].replace({"mercedesbenz": "mercedes"})
            firm_table["is_us"] = firm_table["firm_ids"].str.lower().isin(us_firms)
            base = pd.to_numeric(firm_table["pi0_millions_usd"], errors="coerce").to_numpy(dtype=float)
            delta = pd.to_numeric(firm_table["dpi_millions_usd"], errors="coerce").to_numpy(dtype=float)
            pct = np.full(len(firm_table), np.nan, dtype=float)
            mask = np.isfinite(base) & (base != 0)
            pct[mask] = 100.0 * delta[mask] / base[mask]
            firm_table["pct_change_profit"] = pct
            rows.append(
                firm_table[
                    [
                        "scenario_key",
                        "scenario_label",
                        "label_slug",
                        "firm_ids",
                        "plotted_firm_label",
                        "is_us",
                        "share0_total",
                        "pi0_millions_usd",
                        "pi_cf_millions_usd",
                        "dpi_millions_usd",
                        "pct_change_profit",
                    ]
                ].rename(columns={"share0_total": "market_share"})
            )
        graph_values = pd.concat(rows, ignore_index=True)
    pct_x = graph_values.loc[graph_values["label_slug"] == "vehicles_only_tariff__no_subsidy"].copy()
    pct_y = graph_values.loc[graph_values["label_slug"] == "parts_and_vehicles_tariff__no_subsidy"].copy()
    if pct_x.empty or pct_y.empty:
        raise ValueError("Missing no-subsidy tariff scenarios in profit_changes_graph_values.csv.")

    merged = pct_x[
        ["firm_ids", "plotted_firm_label", "is_us", "market_share", "pct_change_profit", "dpi_millions_usd"]
    ].rename(columns={"pct_change_profit": "pct_x", "dpi_millions_usd": "usd_x"})
    merged = merged.merge(
        pct_y[["firm_ids", "pct_change_profit", "dpi_millions_usd"]].rename(
            columns={"pct_change_profit": "pct_y", "dpi_millions_usd": "usd_y"}
        ),
        on="firm_ids",
        how="inner",
        validate="one_to_one",
    )
    merged = merged.sort_values("market_share", ascending=False).reset_index(drop=True)
    major_labels = set(merged.head(12)["firm_ids"])

    def bubble_size(share: pd.Series) -> np.ndarray:
        return 4000.0 * np.sqrt(np.clip(pd.to_numeric(share, errors="coerce").to_numpy(dtype=float), 0, None))

    def render_scatter(xcol: str, ycol: str, out_name: str, label_selector: set[str], xlabel: str, ylabel: str) -> None:
        fig, ax = plt.subplots(figsize=(10.5, 7.0))
        colors = np.where(merged["is_us"], "#d62728", "#1f77b4")
        ax.scatter(
            merged[xcol],
            merged[ycol],
            s=bubble_size(merged["market_share"]),
            c=colors,
            alpha=0.75,
            edgecolors="white",
            linewidths=0.8,
        )
        ax.axhline(0.0, color="#444444", linewidth=1.0, linestyle="--")
        ax.axvline(0.0, color="#444444", linewidth=1.0, linestyle="--")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2, linewidth=0.6)

        for idx, row in merged.iterrows():
            if row["firm_ids"] not in label_selector:
                continue
            dx = 6 if idx % 2 == 0 else -6
            dy = 6 if idx % 3 == 0 else -8
            ax.annotate(
                str(row["plotted_firm_label"]).title(),
                (row[xcol], row[ycol]),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=9,
            )

        handles = [
            plt.Line2D([0], [0], marker="o", color="w", label="US-headquartered", markerfacecolor="#d62728", markersize=9),
            plt.Line2D([0], [0], marker="o", color="w", label="Non-US-headquartered", markerfacecolor="#1f77b4", markersize=9),
        ]
        ax.legend(handles=handles, loc="best", frameon=False)
        fig.tight_layout()
        fig.savefig(paths.graphs_dir / out_name, dpi=300, bbox_inches="tight")
        plt.close(fig)

    render_scatter(
        "pct_x",
        "pct_y",
        "z_scatter_pct_more.png",
        major_labels,
        "C1: vehicles-only tariff profit change (%)",
        "C2: parts and vehicles tariff profit change (%)",
    )
    render_scatter(
        "pct_x",
        "pct_y",
        "z_scatter_pct_less.png",
        set(merged["firm_ids"]) - major_labels,
        "C1: vehicles-only tariff profit change (%)",
        "C2: parts and vehicles tariff profit change (%)",
    )
    render_scatter(
        "usd_x",
        "usd_y",
        "z_scatter_usd_more.png",
        major_labels,
        "C1: vehicles-only tariff profit change (million USD)",
        "C2: parts and vehicles tariff profit change (million USD)",
    )


def write_references_bbl(paths: PaperPaths, out_name: str = "references.bbl") -> None:
    (paths.generated_dir / out_name).write_text(CANONICAL_BBL)


def normalize_pdf_text(text: str) -> str:
    normalized = text
    replacements = {
        "\u2013": "-",
        "\u2014": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\ufb01": "fi",
        "\ufb02": "fl",
        "\xa0": " ",
        "\u2217": "*",
    }
    for src, dst in replacements.items():
        normalized = normalized.replace(src, dst)
    normalized = normalized.replace("∗", "*")
    normalized = normalized.replace("`", "")
    normalized = unicodedata.normalize("NFKD", normalized)
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = re.sub(r"URL\s*(?=https?://)", "", normalized, flags=re.I)
    normalized = re.sub(r"(?<=\w)-\s+(?=[a-z])", "", normalized)
    normalized = re.sub(r"(?<=:)\s*(?=\S)", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def strip_tex_comments(tex: str) -> str:
    if r"\end{document}" in tex:
        tex = tex.split(r"\end{document}", 1)[0] + r"\end{document}"
    tex = re.sub(r"\\begin\{comment\}.*?\\end\{comment\}", "", tex, flags=re.S)
    lines = []
    for raw_line in tex.splitlines():
        lines.append(re.sub(r"(?<!\\)%.*$", "", raw_line))
    return "\n".join(lines)


def parse_required_assets(tex_path: Path) -> list[str]:
    tex = strip_tex_comments(tex_path.read_text())
    refs = re.findall(r"\\(?:input|includegraphics)(?:\[[^\]]*\])?\{([^}]+)\}", tex)
    return [ref for ref in refs if not ref.startswith("#")]


def validate_generated_assets(paths: PaperPaths) -> dict[str, object]:
    missing: list[str] = []
    for ref in parse_required_assets(paths.canonical_tex):
        candidate = paths.paper_dir / ref
        if candidate.suffix:
            if not candidate.exists():
                missing.append(ref)
            continue
        candidates = [candidate.with_suffix(".tex"), candidate.with_suffix(".png"), candidate]
        if not any(path.exists() for path in candidates):
            missing.append(ref)
    return {"missing_assets": missing, "asset_check_ok": not missing}
