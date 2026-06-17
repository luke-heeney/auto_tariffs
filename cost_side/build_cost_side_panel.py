from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from country_normalization import (  # noqa: E402
    extract_raw_percent_country_pairs,
    extract_percent_country_pairs,
    normalize_country_name,
    normalize_country_series,
    unmapped_country_values,
)

import re

PCT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")
PRODUCT_ID_RE = re.compile(r"^(?P<year>\d{4})_(?P<make_model>.+)$")


def repo_root() -> Path:
    return REPO_ROOT


def cost_side_dir() -> Path:
    return repo_root() / "cost_side"


def outputs_dir() -> Path:
    out = cost_side_dir() / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _extract_pairs(value: object) -> list[tuple[float, str]]:
    if value is None or pd.isna(value):
        return []
    return extract_percent_country_pairs(value)


def _get_n(items: list[tuple[float, str]], idx: int, item_idx: int) -> object:
    try:
        return items[idx][item_idx]
    except Exception:
        return pd.NA


def clean_pcusca(value: object) -> object:
    if pd.isna(value):
        return pd.NA
    text = str(value).replace("％", "%")
    match = PCT_RE.search(text)
    if not match:
        return pd.NA
    return float(match.group(1))


def tidy_country_code(series: pd.Series) -> pd.Series:
    return normalize_country_series(series)


def load_manufacturer_sourcing() -> pd.DataFrame:
    data_path = repo_root() / "processed_data" / "auto_sourcing" / "percentage_data"
    frames: list[pd.DataFrame] = []
    for year in range(2016, 2025):
        file_path = data_path / f"{year}_manuf1.csv"
        if file_path.exists():
            frames.append(pd.read_csv(file_path))
    if not frames:
        raise FileNotFoundError(f"No manufacturer sourcing files found in {data_path}")

    data = pd.concat(frames, ignore_index=True)
    data = data.dropna(subset=["pcUSCA_content"])
    data = data[data["pcUSCA_content"] != " "]
    data = data[data["pcUSCA_content"] != "G (8HP75 & 8HP75PH)"]
    data = data.rename(columns={"year_make_model": "product_ids"})
    data["market_year"] = pd.to_numeric(data["product_ids"].astype(str).str[:4], errors="coerce").astype("Int64")
    return data


def parse_sourcing_content(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    out["pcUSCA_pct"] = out["pcUSCA_content"].apply(clean_pcusca).astype("Float64")

    for col in ["pcOth1_content", "pcOth2_content"]:
        pairs = out[col].apply(_extract_pairs)
        base = col.replace("_content", "")
        out[f"{base}_pct1"] = pairs.apply(lambda x: _get_n(x, 0, 0)).astype("Float64")
        out[f"{base}_code1"] = pairs.apply(lambda x: _get_n(x, 0, 1)).astype("string")
        out[f"{base}_pct2"] = pairs.apply(lambda x: _get_n(x, 1, 0)).astype("Float64")
        out[f"{base}_code2"] = pairs.apply(lambda x: _get_n(x, 1, 1)).astype("string")

    for col in ["pcOth1_pct1", "pcOth2_pct1", "pcOth1_pct2", "pcOth2_pct2", "pcUSCA_pct"]:
        out[col] = pd.to_numeric(out[col], errors="coerce") / 100.0

    for code_col in ["pcOth1_code1", "pcOth1_code2", "pcOth2_code1", "pcOth2_code2"]:
        out[code_col] = tidy_country_code(out[code_col].astype("string"))

    return out.drop(columns=["pcUSCA_content", "pcOth1_content", "pcOth2_content"])


def write_country_parsing_diagnostics(data: pd.DataFrame, out_dir: Path) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    unmapped_inputs: list[str] = []
    for source_col in ["pcOth1_content", "pcOth2_content"]:
        if source_col not in data.columns:
            continue
        for raw_value in data[source_col].dropna():
            for pct, raw_country, canonical_country in extract_raw_percent_country_pairs(raw_value):
                rows.append(
                    {
                        "source_column": source_col,
                        "raw_value": raw_value,
                        "pct": pct / 100.0,
                        "raw_country": raw_country,
                        "canonical_country": canonical_country,
                    }
                )
                if canonical_country is None:
                    unmapped_inputs.append(raw_country)

    detail = pd.DataFrame(rows)
    detail.to_csv(out_dir / "country_parsing_raw_values.csv", index=False)

    if detail.empty:
        summary = pd.DataFrame(columns=["raw_country", "canonical_country", "rows", "mean_pct"])
    else:
        summary = (
            detail.groupby(["raw_country", "canonical_country"], dropna=False)
            .agg(rows=("raw_value", "size"), mean_pct=("pct", "mean"))
            .reset_index()
            .sort_values(["canonical_country", "raw_country"])
        )
    summary.to_csv(out_dir / "country_parsing_summary.csv", index=False)

    return {
        "parsed_country_tokens": int(detail.shape[0]),
        "parsed_country_unique_raw_tokens": int(detail["raw_country"].nunique()) if not detail.empty else 0,
        "parsed_country_unique_canonical": int(detail["canonical_country"].nunique()) if not detail.empty else 0,
        "unmapped_country_tokens": "; ".join(unmapped_country_values(unmapped_inputs)),
        "unmapped_country_token_count": int(len(unmapped_country_values(unmapped_inputs))),
    }


def load_blp_data() -> pd.DataFrame:
    path = repo_root() / "post_est" / "data" / "raw" / "blpUS0804.csv"
    blp = pd.read_csv(path)
    blp["market_year"] = pd.to_numeric(blp["market_year"], errors="coerce").astype("Int64")
    blp["product_ids"] = blp["product_ids"].astype(str)
    blp = blp[blp["product_ids"].str[:4] == blp["market_year"].astype(str)]
    return blp


def collapse_product_rows(df: pd.DataFrame) -> pd.DataFrame:
    oth_pct_cols = ["pcOth1_pct1", "pcOth1_pct2", "pcOth2_pct1", "pcOth2_pct2"]
    oth_code_cols = ["pcOth1_code1", "pcOth1_code2", "pcOth2_code1", "pcOth2_code2"]
    oth_pairs = list(zip(oth_code_cols, oth_pct_cols))
    meta_cols = ["assembly1", "market_year", "plant_country", "vehicle_type"]
    na_like = {"<NA>", "NA", "NaN", "None", ""}
    out_rows: list[dict[str, object]] = []

    def mode_or_first(series: pd.Series) -> object:
        series = series.dropna()
        if series.empty:
            return np.nan
        counts = Counter(series)
        winners = [value for value, count in counts.items() if count == counts.most_common(1)[0][1]]
        for value in series:
            if value in winners:
                return value
        return series.iloc[0]

    work = df.copy()
    for col in oth_code_cols + ["assembly1", "plant_country"]:
        if col in work.columns:
            work[col] = work[col].where(~work[col].astype(str).isin(na_like), np.nan)
    for col in ["pcUSCA_pct"] + oth_pct_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work["product_ids"] = work["product_ids"].astype(str)

    for product_id, group in work.groupby("product_ids", dropna=False):
        has_any_foreign = group[oth_pct_cols].notna().any(axis=1)
        kept = group.loc[has_any_foreign].copy() if has_any_foreign.any() else group.copy()

        direct_primary = kept.loc[
            kept["pcOth1_code1"].notna() & kept["pcOth1_pct1"].notna(),
            ["pcOth1_code1", "pcOth1_pct1"],
        ].copy()
        unique_direct_primary_countries = sorted(
            {str(country) for country in direct_primary["pcOth1_code1"].dropna()}
        )

        country_sums: defaultdict[str, float] = defaultdict(float)
        country_counts: defaultdict[str, int] = defaultdict(int)
        for _, row in kept.iterrows():
            for code_col, pct_col in oth_pairs:
                country = row.get(code_col)
                pct = row.get(pct_col)
                if pd.notna(country) and pd.notna(pct):
                    country_sums[str(country)] += float(pct)
                    country_counts[str(country)] += 1

        foreign_avgs = [
            (country, country_sums[country] / country_counts[country])
            for country in country_sums
            if country_counts[country] > 0
        ]
        foreign_avgs.sort(key=lambda x: (-x[1], x[0]))

        fill_order = [
            ("pcOth1_code1", "pcOth1_pct1"),
            ("pcOth1_code2", "pcOth1_pct2"),
            ("pcOth2_code1", "pcOth2_pct1"),
            ("pcOth2_code2", "pcOth2_pct2"),
        ]
        filled: dict[str, object] = {col: np.nan for col in oth_code_cols + oth_pct_cols}
        for (code_col, pct_col), (country, avg) in zip(fill_order, foreign_avgs):
            filled[code_col] = country
            filled[pct_col] = float(avg)

        rep: dict[str, object] = {
            "product_ids": product_id,
            "pcUSCA_pct": kept["pcUSCA_pct"].mean(skipna=True) if "pcUSCA_pct" in kept else np.nan,
            "raw_row_count": int(group.shape[0]),
            "raw_rows_with_any_foreign_content": int(has_any_foreign.sum()),
            "raw_primary_pair_count": int(direct_primary.shape[0]),
            "raw_primary_country_count": int(len(unique_direct_primary_countries)),
            "raw_primary_countries": " | ".join(unique_direct_primary_countries),
            "uses_collapsed_duplicate_row": bool(group.shape[0] > 1),
        }
        rep.update(filled)

        for col in meta_cols:
            if col in kept.columns:
                rep[col] = mode_or_first(kept[col])

        out_rows.append(rep)

    ordered_cols = [
        "product_ids",
        "assembly1",
        "market_year",
        "plant_country",
        "vehicle_type",
        "pcUSCA_pct",
        "pcOth1_pct1",
        "pcOth1_code1",
        "pcOth1_pct2",
        "pcOth1_code2",
        "pcOth2_pct1",
        "pcOth2_code1",
        "pcOth2_pct2",
        "pcOth2_code2",
        "raw_row_count",
        "raw_rows_with_any_foreign_content",
        "raw_primary_pair_count",
        "raw_primary_country_count",
        "raw_primary_countries",
        "uses_collapsed_duplicate_row",
    ]
    collapsed = pd.DataFrame(out_rows)
    ordered = [col for col in ordered_cols if col in collapsed.columns]
    trailing = [col for col in collapsed.columns if col not in ordered]
    return collapsed[ordered + trailing]


def attach_panel_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    extracted = out["product_ids"].astype(str).str.extract(PRODUCT_ID_RE)
    out["year"] = pd.to_numeric(extracted["year"], errors="coerce").astype("Int64")
    out["make_model"] = extracted["make_model"]
    return out


def filter_analysis_sample(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["pcOth1_pct1"] = pd.to_numeric(out["pcOth1_pct1"], errors="coerce")
    out = out[out["pcOth1_pct1"] > 0].copy()
    out = out.groupby("make_model", dropna=False).filter(lambda g: g["year"].nunique() >= 2)
    return out.loc[:, ~out.columns.str.startswith("pcOth2")].copy()


def merge_costs_and_chars(df: pd.DataFrame, blp: pd.DataFrame) -> pd.DataFrame:
    costs_path = repo_root() / "post_est" / "data" / "derived" / "vehicle_costs_markups_chars.csv"
    costs = pd.read_csv(costs_path)
    costs = costs[["product_ids", "market_year", "costs", "markups", "engine_type", "ev", "hybrid"]].copy()
    costs["market_year"] = pd.to_numeric(costs["market_year"], errors="coerce").astype("Int64")
    costs["product_ids"] = costs["product_ids"].astype(str)
    costs = costs[(costs["market_year"] >= 2016) & (costs["market_year"] <= 2024)]
    costs = costs[costs["product_ids"].str[:4] == costs["market_year"].astype(str)]
    costs = costs.dropna(subset=["product_ids", "market_year"]).drop_duplicates(
        subset=["product_ids", "market_year"],
        keep="last",
    )

    out = df.merge(
        costs,
        on=["product_ids", "market_year"],
        how="inner",
        validate="many_to_one",
    )

    chars = blp[["product_ids", "market_year", "size", "weight", "hp", "mpg"]].copy()
    chars = chars.drop_duplicates(subset=["product_ids", "market_year", "size", "weight", "hp", "mpg"])
    out = out.merge(chars, on=["product_ids", "market_year"], how="left")
    return out


def load_exchange_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    imports_path = repo_root() / "processed_data" / "auto_sourcing" / "parts_imports_countries.csv"
    exchange_path = repo_root() / "processed_data" / "exchange_rates" / "exchange_rates.csv"

    imports = pd.read_csv(imports_path).rename(columns={"Partner": "country"}).head(30)
    imports = imports.drop(columns=["2008", "2009", "2010", "2011", "2012", "2013"])
    imports["country"] = normalize_country_series(imports["country"])
    imports = imports.dropna(subset=["country"]).copy()

    exchange = pd.read_csv(exchange_path)
    exchange = exchange.drop(columns=["Country Code", "2010", "2011", "2012", "2013"])
    exchange = exchange.rename(columns={"Country Name": "country"})
    exchange["country"] = normalize_country_series(exchange["country"])

    cleaned_imports = imports.copy()
    num_cols = [col for col in cleaned_imports.columns if col != "country"]
    cleaned_imports[num_cols] = (
        cleaned_imports[num_cols]
        .replace({r"[,\s]": ""}, regex=True)
        .replace({"": np.nan})
        .apply(pd.to_numeric, errors="coerce")
        .astype("Int64")
    )
    cleaned_imports = cleaned_imports.groupby("country", as_index=False)[num_cols].sum(min_count=1)

    exchange = exchange.groupby("country", as_index=False).first()
    return cleaned_imports, exchange


def normalize_exchange_to_2015(exchange: pd.DataFrame) -> pd.DataFrame:
    out = exchange.copy()
    year_cols = [col for col in out.columns if col != "country"]
    if "2015" not in out.columns:
        raise ValueError("Exchange-rate file is missing the 2015 normalization column.")
    out[year_cols] = out[year_cols].apply(pd.to_numeric, errors="coerce")
    denom = out["2015"].replace({0: np.nan})
    out[year_cols] = out[year_cols].div(denom, axis=0).astype(float)
    return out


def exchange_long(exchange: pd.DataFrame) -> pd.DataFrame:
    long = exchange.melt(id_vars="country", var_name="year", value_name="rer")
    long["year"] = pd.to_numeric(long["year"], errors="coerce").astype("Int64")
    long["rer"] = pd.to_numeric(long["rer"], errors="coerce")
    return long


def add_rer_column(df: pd.DataFrame, exchange_norm: pd.DataFrame, code_col: str, out_col: str) -> pd.DataFrame:
    out = df.copy()
    out["_year_join"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    merged = out.merge(
        exchange_long(exchange_norm),
        left_on=[code_col, "_year_join"],
        right_on=["country", "year"],
        how="left",
    )
    out[out_col] = merged["rer"].values
    return out.drop(columns=["_year_join"])


def build_trade_weighted_rer(cleaned_imports: pd.DataFrame, exchange_norm: pd.DataFrame) -> pd.DataFrame:
    imp_long = cleaned_imports.melt(id_vars="country", var_name="year", value_name="imports")
    ex_long = exchange_norm.melt(id_vars="country", var_name="year", value_name="rer")

    imp_long["year"] = pd.to_numeric(imp_long["year"], errors="coerce").astype("Int64")
    ex_long["year"] = pd.to_numeric(ex_long["year"], errors="coerce").astype("Int64")
    imp_long["imports"] = pd.to_numeric(imp_long["imports"], errors="coerce")
    ex_long["rer"] = pd.to_numeric(ex_long["rer"], errors="coerce")

    merged = imp_long.merge(ex_long, on=["country", "year"], how="inner")
    merged["weighted"] = merged["imports"] * merged["rer"]

    agg = merged.groupby("year", as_index=False).agg(
        weighted_sum=("weighted", "sum"),
        import_sum_matched=("imports", "sum"),
        countries_used=("country", "nunique"),
    )
    totals = imp_long.groupby("year", as_index=False).agg(total_imports=("imports", "sum"))
    out = agg.merge(totals, on="year", how="left")
    out["tw_rer_index"] = out["weighted_sum"] / out["import_sum_matched"]
    out["coverage_share"] = out["import_sum_matched"] / out["total_imports"]
    return out


def add_panel_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["plant_country"] = normalize_country_series(out["plant_country"]).astype("string")
    out["is_us_assembled"] = out["plant_country"].eq("United States")
    out["is_foreign_assembled"] = out["plant_country"].notna() & out["plant_country"].ne("United States")

    switch_counts = (
        out.groupby("make_model", dropna=False)["pcOth1_code1"]
        .nunique(dropna=True)
        .rename("source_country_switch_count")
    )
    out = out.merge(switch_counts, on="make_model", how="left")
    out["source_country_switch_count"] = (
        pd.to_numeric(out["source_country_switch_count"], errors="coerce").fillna(0).astype(int)
    )
    out["source_country_stable"] = out["source_country_switch_count"] <= 1
    out["has_direct_primary_country"] = out["raw_primary_country_count"] > 0
    out["primary_source_country_consistent"] = out["raw_primary_country_count"] <= 1
    out["canonical_exposure_ok"] = (
        out["source_country_stable"]
        & out["has_direct_primary_country"]
        & out["primary_source_country_consistent"]
    )
    return out


def domestic_panel_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return the slim canonical domestic panel used by regression scripts."""
    drop_cols = [
        "assembly1",
        "raw_row_count",
        "raw_rows_with_any_foreign_content",
        "raw_primary_pair_count",
        "raw_primary_country_count",
        "raw_primary_countries",
        "uses_collapsed_duplicate_row",
        "is_us_assembled",
        "is_foreign_assembled",
        "source_country_switch_count",
        "source_country_stable",
        "has_direct_primary_country",
        "primary_source_country_consistent",
        "canonical_exposure_ok",
    ]
    return df.drop(columns=[col for col in drop_cols if col in df.columns])


def write_tidy_metrics(metrics: dict[str, object], path: Path) -> None:
    rows = [{"metric": key, "value": value} for key, value in metrics.items()]
    pd.DataFrame(rows).to_csv(path, index=False)


def write_panel_summary_markdown(
    all_panel: pd.DataFrame,
    sample_counts: pd.DataFrame,
    diagnostics: dict[str, object],
    switchers: pd.DataFrame,
    path: Path,
) -> None:
    lines = [
        "# Panel Build Summary",
        "",
        "This file is generated by `cost_side/build_cost_side_panel.py`.",
        "",
        "## Sample Counts",
        "",
    ]
    for row in sample_counts.itertuples(index=False):
        lines.append(
            f"- `{row.sample}`: {int(row.rows)} rows, {int(row.make_models)} make-models"
        )
    lines.extend(
        [
            "",
            "## Diagnostics",
            "",
        ]
    )
    for key, value in diagnostics.items():
        lines.append(f"- `{key}`: {value}")

    if not switchers.empty:
        lines.extend(
            [
                "",
                "## Source-Country Switchers",
                "",
                f"- Count of switcher make-models: {switchers['make_model'].nunique()}",
                f"- Rows in switcher make-models: {switchers['row_count'].sum()}",
            ]
        )

    lines.extend(
        [
            "",
            "## Flag Definitions Used In This Summary",
            "",
            "- `source_country_stable`: at most one observed `pcOth1_code1` within a make-model after the exchange-rate merge.",
            "- `primary_source_country_consistent`: all directly observed raw primary source countries agree within the product-year.",
            "- `canonical_exposure_ok`: stable source country plus a directly observed and source-country-consistent raw primary source.",
        ]
    )

    path.write_text("\n".join(lines) + "\n")


def build_and_write_panels() -> None:
    out_dir = outputs_dir()
    manufacturer = load_manufacturer_sourcing()
    country_diag = write_country_parsing_diagnostics(manufacturer, out_dir)
    parsed = parse_sourcing_content(manufacturer)
    blp = load_blp_data()

    plant_country = blp[["product_ids", "market_year", "plant_country"]].copy()
    plant_country["plant_country"] = normalize_country_series(plant_country["plant_country"])
    plant_country = plant_country.drop_duplicates(subset=["product_ids", "market_year", "plant_country"])
    merged = parsed.merge(
        plant_country,
        on=["product_ids", "market_year"],
        how="left",
        validate="many_to_one",
    )
    merged = merged.dropna(subset=["plant_country"]).copy()
    collapsed = collapse_product_rows(merged)
    keyed = attach_panel_keys(collapsed)
    analysis = filter_analysis_sample(keyed)
    analysis = merge_costs_and_chars(analysis, blp)

    imports, exchange = load_exchange_inputs()
    exchange_norm = normalize_exchange_to_2015(exchange)
    analysis = add_rer_column(analysis, exchange_norm, "pcOth1_code1", "rer_pcOth1_code1_n2015")
    analysis = add_rer_column(analysis, exchange_norm, "pcOth1_code2", "rer_pcOth1_code2_n2015")

    pre_drop = analysis.copy()
    analysis = analysis.dropna(subset=["rer_pcOth1_code1_n2015"]).copy()
    tw_rer = build_trade_weighted_rer(imports, exchange_norm)
    analysis = analysis.merge(tw_rer[["year", "tw_rer_index"]], on="year", how="left")
    analysis["costs"] = pd.to_numeric(analysis["costs"], errors="coerce") * 10.0
    analysis["make"] = analysis["make_model"].astype(str).str.split("_").str[0]
    analysis = add_panel_flags(analysis)
    analysis = analysis.sort_values(["make_model", "year", "product_ids"]).reset_index(drop=True)

    all_panel_path = cost_side_dir() / "cost_side_panel_all.csv"
    us_panel_path = cost_side_dir() / "cost_side_panel.csv"
    foreign_panel_path = cost_side_dir() / "cost_side_panel_foreign.csv"

    analysis.to_csv(all_panel_path, index=False)
    canonical_domestic = analysis.loc[analysis["is_us_assembled"] & analysis["canonical_exposure_ok"]].copy()
    domestic_panel_columns(canonical_domestic).to_csv(us_panel_path, index=False)
    analysis.loc[analysis["is_foreign_assembled"]].to_csv(foreign_panel_path, index=False)

    sample_specs = [
        ("all", analysis),
        ("us_all", analysis.loc[analysis["is_us_assembled"]]),
        ("canonical_domestic", canonical_domestic),
        ("foreign_all", analysis.loc[analysis["is_foreign_assembled"]]),
        (
            "foreign_source_stable",
            analysis.loc[analysis["is_foreign_assembled"] & analysis["source_country_stable"]],
        ),
    ]
    sample_counts = pd.DataFrame(
        [
            {
                "sample": name,
                "rows": frame.shape[0],
                "make_models": frame["make_model"].nunique(),
            }
            for name, frame in sample_specs
        ]
    )
    sample_counts.to_csv(out_dir / "panel_build_sample_counts.csv", index=False)

    switchers = (
        analysis.groupby("make_model", dropna=False)
        .agg(
            source_country_switch_count=("source_country_switch_count", "max"),
            source_countries=("pcOth1_code1", lambda s: " | ".join(sorted({str(v) for v in s.dropna()}))),
            row_count=("product_ids", "size"),
            is_us_assembled=("is_us_assembled", "max"),
            is_foreign_assembled=("is_foreign_assembled", "max"),
        )
        .reset_index()
    )
    switchers = switchers.loc[switchers["source_country_switch_count"] > 1].copy()
    switchers.to_csv(out_dir / "source_country_switchers.csv", index=False)

    primary_country_conflicts = analysis.loc[
        analysis["is_us_assembled"]
        & analysis["source_country_stable"]
        & analysis["has_direct_primary_country"]
        & ~analysis["primary_source_country_consistent"],
        [
            "product_ids",
            "make_model",
            "year",
            "plant_country",
            "raw_row_count",
            "raw_primary_pair_count",
            "raw_primary_country_count",
            "raw_primary_countries",
            "pcOth1_code1",
            "pcOth1_pct1",
        ],
    ].copy()
    primary_country_conflicts.to_csv(out_dir / "primary_source_country_conflicts.csv", index=False)

    diagnostics = {
        "raw_manufacturer_rows": int(manufacturer.shape[0]),
        **country_diag,
        "rows_after_plant_country_merge": int(merged.shape[0]),
        "rows_after_product_collapse": int(collapsed.shape[0]),
        "rows_after_foreign_content_filter": int(analysis.shape[0]),
        "pre_drop_missing_rer_rows": int(pre_drop["rer_pcOth1_code1_n2015"].isna().sum()),
        "switcher_make_models": int(switchers["make_model"].nunique()),
        "rows_with_duplicate_collapse": int(analysis["uses_collapsed_duplicate_row"].sum()),
        "rows_with_direct_primary_country": int(analysis["has_direct_primary_country"].sum()),
        "rows_with_primary_source_country_conflict": int(primary_country_conflicts.shape[0]),
        "rows_in_canonical_domestic_sample": int(canonical_domestic.shape[0]),
    }
    write_tidy_metrics(diagnostics, out_dir / "panel_build_diagnostics.csv")
    write_panel_summary_markdown(
        analysis,
        sample_counts,
        diagnostics,
        switchers,
        out_dir / "panel_build_summary.md",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build reproducible cost-side panels and diagnostics.")
    parser.parse_args()
    build_and_write_panels()


if __name__ == "__main__":
    main()
