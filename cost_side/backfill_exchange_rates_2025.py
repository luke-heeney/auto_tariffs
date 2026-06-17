#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from io import StringIO
from pathlib import Path

import pandas as pd
import requests


REPO_ROOT = Path(__file__).resolve().parents[1]
EXCHANGE_PATH = REPO_ROOT / "processed_data" / "exchange_rates" / "exchange_rates.csv"
IMPORTS_PATH = REPO_ROOT / "processed_data" / "auto_sourcing" / "parts_imports_countries.csv"
SUMMARY_MD_PATH = REPO_ROOT / "cost_side" / "outputs" / "exchange_rate_2025_backfill_summary.md"
SUMMARY_CSV_PATH = REPO_ROOT / "cost_side" / "outputs" / "exchange_rate_2025_backfill_values.csv"

REPORT_BASE_URL = "https://www.imf.org/external/np/fin/ert/gui/pages"
YEAR = 2025
CACHE_REQUIRED_COLUMNS = {
    "country",
    "year",
    "value",
    "formatted_value",
    "imf_code",
    "imf_label",
    "inverted",
}


@dataclass(frozen=True)
class SeriesSpec:
    imf_code: str
    label_prefix: str
    repo_countries: tuple[str, ...]
    invert: bool = False


SERIES_SPECS = (
    SeriesSpec("BRA", "Brazilian real", ("Brazil",)),
    SeriesSpec("CHL", "Chilean peso", ("Chile",)),
    SeriesSpec("CHN", "Chinese yuan", ("China",)),
    SeriesSpec("CZE", "Czech koruna", ("Czechia",)),
    SeriesSpec(
        "EMU",
        "Euro",
        (
            "Austria",
            "France",
            "Germany",
            "Italy",
            "Netherlands",
            "Portugal",
            "Slovakia",
            "Spain",
        ),
        invert=True,
    ),
    SeriesSpec("GBR", "U.K. pound", ("United Kingdom",), invert=True),
    SeriesSpec("IND", "Indian rupee", ("India",)),
    SeriesSpec("JPN", "Japanese yen", ("Japan",)),
    SeriesSpec("KOR", "Korean won", ("Korea",)),
    SeriesSpec("MEX", "Mexican peso", ("Mexico",)),
    SeriesSpec("MYS", "Malaysian ringgit", ("Malaysia",)),
    SeriesSpec("PHL", "Philippine peso", ("Philippines",)),
    SeriesSpec("POL", "Polish zloty", ("Poland",)),
    SeriesSpec("SWE", "Swedish krona", ("Sweden",)),
    SeriesSpec("THA", "Thai baht", ("Thailand",)),
)


def dotnet_ticks(date_str: str) -> int:
    base = datetime(1, 1, 1)
    current = datetime.strptime(date_str, "%Y-%m-%d")
    return (current - base).days * 24 * 60 * 60 * 10_000_000


def fetch_report_tsv(codes: list[str], year: int) -> str:
    ct = ",".join(f"'{code}'" for code in codes)
    fr = dotnet_ticks(f"{year}-01-01")
    to = dotnet_ticks(f"{year}-12-31")
    report_url = (
        f"{REPORT_BASE_URL}/Report.aspx"
        f"?CT={ct}&EX=REP&P=DateRange&Fr={fr}&To={to}"
        "&CF=Compressed&CUF=Period&DS=Ascending&DT=Blank"
    )
    session = requests.Session()
    report_response = session.get(report_url, timeout=60)
    report_response.raise_for_status()

    data_response = session.get(f"{REPORT_BASE_URL}/ReportData.aspx?Type=TSV", timeout=60)
    data_response.raise_for_status()
    return data_response.text


def parse_report_tsv(text: str) -> pd.DataFrame:
    lines = [line for line in text.splitlines() if line.strip()]
    header_index = next(
        idx
        for idx, line in enumerate(lines)
        if line.lstrip().startswith("Date\t")
    )
    df = pd.read_csv(StringIO("\n".join(lines[header_index:])), sep="\t")
    df.columns = [col.strip().replace("\xa0", "") for col in df.columns]

    clean_cols = [col for col in df.columns if col and not col.startswith("Unnamed:")]
    df = df[clean_cols]
    for col in df.columns:
        if col == "Date":
            continue
        df[col] = pd.to_numeric(
            df[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("\xa0", "", regex=False)
            .str.strip(),
            errors="coerce",
        )
    return df


def annual_series_values(df: pd.DataFrame, specs: tuple[SeriesSpec, ...]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for spec in specs:
        matching_cols = [col for col in df.columns if col.startswith(spec.label_prefix)]
        if not matching_cols:
            raise KeyError(f"No IMF report column matched prefix '{spec.label_prefix}'.")
        raw_mean = df[matching_cols[0]].mean(skipna=True)
        if pd.isna(raw_mean):
            raise ValueError(f"Series '{matching_cols[0]}' has no usable observations.")
        value = 1.0 / raw_mean if spec.invert else float(raw_mean)
        for country in spec.repo_countries:
            rows.append(
                {
                    "country": country,
                    "year": YEAR,
                    "value": value,
                    "formatted_value": f"{value:.9f}".rstrip("0").rstrip("."),
                    "imf_code": spec.imf_code,
                    "imf_label": matching_cols[0],
                    "inverted": spec.invert,
                }
            )
    return pd.DataFrame(rows).sort_values("country").reset_index(drop=True)


def load_cached_values(year: int) -> pd.DataFrame:
    if not SUMMARY_CSV_PATH.exists():
        raise FileNotFoundError(
            f"Cached exchange-rate backfill values not found at {SUMMARY_CSV_PATH}. "
            "Run `python cost_side/backfill_exchange_rates_2025.py --refresh` when "
            "network access is available to regenerate the cache."
        )

    values_df = pd.read_csv(SUMMARY_CSV_PATH)
    missing_columns = sorted(CACHE_REQUIRED_COLUMNS.difference(values_df.columns))
    if missing_columns:
        raise ValueError(
            f"Cached exchange-rate backfill is missing required columns: {missing_columns}"
        )

    values_df = values_df.copy()
    values_df["year"] = pd.to_numeric(values_df["year"], errors="raise").astype(int)
    values_df = values_df[values_df["year"].eq(year)].reset_index(drop=True)
    if values_df.empty:
        raise ValueError(
            f"Cached exchange-rate backfill has no rows for {year}: {SUMMARY_CSV_PATH}"
        )

    values_df["value"] = pd.to_numeric(values_df["value"], errors="raise")
    values_df["formatted_value"] = (
        values_df["formatted_value"]
        .fillna(values_df["value"].map(lambda value: f"{value:.9f}".rstrip("0").rstrip(".")))
        .astype(str)
    )
    values_df["inverted"] = values_df["inverted"].astype(str).str.lower().isin(
        {"true", "1", "yes"}
    )
    return values_df.sort_values("country").reset_index(drop=True)


def update_exchange_csv(values_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    exchange = pd.read_csv(EXCHANGE_PATH, dtype=str, encoding="utf-8-sig")
    if "2025" not in exchange.columns:
        exchange["2025"] = ""

    updated_countries = []
    for row in values_df.itertuples(index=False):
        mask = exchange["Country Name"] == row.country
        if not mask.any():
            continue
        exchange.loc[mask, "2025"] = row.formatted_value
        updated_countries.append(row.country)

    exchange.to_csv(EXCHANGE_PATH, index=False, encoding="utf-8-sig")
    return exchange, sorted(updated_countries)


def write_summary(
    values_df: pd.DataFrame, updated_countries: list[str], run_source: str
) -> None:
    imports = pd.read_csv(IMPORTS_PATH, encoding="utf-8-sig")
    top30 = imports.iloc[:30, 0].astype(str).tolist()
    filled_top30 = [country for country in top30 if country in updated_countries]
    missing_top30 = [country for country in top30 if country not in updated_countries]

    SUMMARY_MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    values_df.to_csv(SUMMARY_CSV_PATH, index=False)

    direct_future_countries = ["Mexico", "Japan", "Korea", "Germany", "Spain", "Sweden"]
    sample_countries_present = [country for country in direct_future_countries if country in updated_countries]

    lines = [
        "# 2025 Exchange-Rate Backfill",
        "",
        "This file is generated by `cost_side/backfill_exchange_rates_2025.py`.",
        "",
        "## Source",
        "",
        f"- Values source for this run: {run_source}.",
        "- IMF Exchange Rate Report Wizard, representative rates, date range January 1, 2025 to December 31, 2025.",
        "- Query URL template:",
        f"  - `{REPORT_BASE_URL}/Report.aspx?CT='...'&EX=REP&P=DateRange&Fr={dotnet_ticks('2025-01-01')}&To={dotnet_ticks('2025-12-31')}&CF=Compressed&CUF=Period&DS=Ascending&DT=Blank`",
        "- Annual values are the arithmetic mean of the available daily representative rates in the TSV export.",
        "- Series quoted by the IMF as U.S. dollars per currency unit are inverted before writing them into `processed_data/exchange_rates/exchange_rates.csv` so the stored values remain in local-currency-units per U.S. dollar, consistent with the existing file.",
        "",
        "## Countries Updated",
        "",
        *[f"- {row.country}: {row.formatted_value} (`{row.imf_label}`; inverted={row.inverted})" for row in values_df.itertuples(index=False)],
        "",
        "## Top-30 Import Partners",
        "",
        f"- Filled from the IMF representative-rate query: {', '.join(filled_top30)}",
        f"- Still missing 2025 in `exchange_rates.csv`: {', '.join(missing_top30)}",
        "",
        "## Cost-Side Placebo Relevance",
        "",
        "- The 2024 domestic current-sample rows that need `RER_{t+1}` use only these source countries:",
        f"  - {', '.join(sample_countries_present)}",
        "- So the backfill now covers every source country needed for the direct future-RER placebo samples.",
    ]
    SUMMARY_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Populate the 2025 exchange-rate column used by cost-side robustness checks. "
            "By default this uses the checked-in cached IMF backfill values so downstream "
            "replication does not depend on live network access."
        )
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help=(
            "Fetch fresh values from the IMF Exchange Rate Report Wizard and overwrite "
            "the cached backfill CSV before updating exchange_rates.csv."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.refresh:
        report_df = parse_report_tsv(
            fetch_report_tsv([spec.imf_code for spec in SERIES_SPECS], YEAR)
        )
        values_df = annual_series_values(report_df, SERIES_SPECS)
        run_source = "live IMF fetch"
    else:
        values_df = load_cached_values(YEAR)
        run_source = f"cached values in `{SUMMARY_CSV_PATH.relative_to(REPO_ROOT)}`"

    _, updated_countries = update_exchange_csv(values_df)
    write_summary(values_df, updated_countries, run_source)
    print("Using", run_source)
    print("Updated", EXCHANGE_PATH)
    print("Wrote", SUMMARY_CSV_PATH)
    print("Wrote", SUMMARY_MD_PATH)


if __name__ == "__main__":
    main()
