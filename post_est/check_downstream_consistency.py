from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
POST_EST_DIR = REPO_ROOT / "post_est"
CANONICAL_CONFIG = POST_EST_DIR / "results_config.json"
BASELINE_LABEL = "no tariff (no subsidy)"
TOL = 1e-7
PASS_THROUGH_TOL = 5e-5
NONCANONICAL_COUNTRY_LABELS = {"Great Britain", "South Korea", "UK", "GB"}
COST_SIDE_ROBUSTNESS_COEFFS = REPO_ROOT / "cost_side" / "outputs" / "cost_reg_robustness_coefficients.csv"
COST_SIDE_BASELINE_SAMPLE = "canonical_domestic"
COST_SIDE_BASELINE_SPEC = "levels_baseline"
COST_SIDE_BASELINE_TERM = "ln_inv_rer_code1:pcOth1_pct1_lag1"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from country_normalization import normalize_country_series, normalize_country_tariffs  # noqa: E402


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


class Checker:
    def __init__(self) -> None:
        self.results: list[CheckResult] = []

    def check(self, name: str, ok: bool, detail: str) -> None:
        self.results.append(CheckResult(name=name, ok=bool(ok), detail=detail))

    def require(self, name: str, path: Path) -> bool:
        ok = path.exists()
        self.check(name, ok, str(path))
        return ok

    @property
    def ok(self) -> bool:
        return all(r.ok for r in self.results)


def _resolve_from(base_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _load_run_modules():
    if str(POST_EST_DIR) not in sys.path:
        sys.path.insert(0, str(POST_EST_DIR))
    import run_cf_batch  # type: ignore
    from helpers.counterfactual_reporting import default_scenario_specs  # type: ignore

    return run_cf_batch, default_scenario_specs


def _slug(label: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in label).strip("_")


def _numeric_cell(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if " (" in text:
        text = text.split(" (", 1)[0]
    return float(text)


def _abs_leq(value: float, tol: float = TOL) -> bool:
    return math.isfinite(value) and abs(value) <= tol


def _same_float(left: object, right: object, tol: float = TOL) -> bool:
    try:
        return abs(float(left) - float(right)) <= tol
    except Exception:
        return False


def _same_mapping(left: dict[str, Any] | None, right: dict[str, Any] | None) -> bool:
    left = left or {}
    right = right or {}
    if set(left) != set(right):
        return False
    return all(_same_float(left[k], right[k]) for k in left)


def _config_results_path(config: dict[str, Any]) -> Path:
    return _resolve_from(POST_EST_DIR, str(config["results_file"]))


def _baseline_cost_side_pass_through() -> float:
    coeffs = pd.read_csv(COST_SIDE_ROBUSTNESS_COEFFS)
    row = coeffs.loc[
        coeffs["sample"].astype(str).eq(COST_SIDE_BASELINE_SAMPLE)
        & coeffs["spec"].astype(str).eq(COST_SIDE_BASELINE_SPEC)
        & coeffs["coefficient"].astype(str).eq(COST_SIDE_BASELINE_TERM)
    ]
    if row.empty:
        raise ValueError(
            "Could not find canonical baseline cost-side pass-through coefficient "
            f"{COST_SIDE_BASELINE_SAMPLE}/{COST_SIDE_BASELINE_SPEC}/{COST_SIDE_BASELINE_TERM}"
        )
    return float(row.iloc[0]["estimate"])


def locate_canonical_bundle(config: dict[str, Any], override: Path | None = None) -> Path:
    if override is not None:
        return override.resolve()

    results_path = _config_results_path(config)
    candidates: list[tuple[float, Path]] = []
    saved_outputs = POST_EST_DIR / "saved_outputs"
    for bundle_dir in saved_outputs.iterdir():
        if not bundle_dir.is_dir():
            continue
        meta_path = bundle_dir / "metadata.json"
        if not meta_path.exists():
            continue
        try:
            meta = _load_json(meta_path)
        except Exception:
            continue
        if meta.get("reporting_baseline_label") != BASELINE_LABEL:
            continue
        if Path(str(meta.get("results_file", ""))).resolve() != results_path:
            continue
        if bundle_dir.name.endswith("_rebased_b0_rebased_b0"):
            continue
        source_dir = Path(str(meta.get("source_saved_output_dir", "")))
        if not source_dir.is_absolute():
            source_dir = (REPO_ROOT / source_dir).resolve()
        if not source_dir.exists():
            continue
        candidates.append((bundle_dir.stat().st_mtime, bundle_dir))
    if not candidates:
        raise FileNotFoundError("No canonical rebased B0 bundle found under post_est/saved_outputs.")
    return max(candidates, key=lambda item: item[0])[1].resolve()


def validate_config(checker: Checker, config: dict[str, Any]) -> None:
    checker.require("canonical config exists", CANONICAL_CONFIG)
    required = [
        "results_file",
        "product_data_45W",
        "product_data_subsidy",
        "agent_data_45W",
        "agent_data_subsidy",
        "ownership_mode",
        "owner_mapping_path",
        "allow_unmapped_brands",
        "parts_cost_adjustment",
    ]
    missing = [key for key in required if key not in config]
    checker.check("canonical config required keys", not missing, f"missing={missing}")

    path_keys = [
        "results_file",
        "product_data_45W",
        "product_data_subsidy",
        "agent_data_45W",
        "agent_data_subsidy",
        "owner_mapping_path",
    ]
    for key in path_keys:
        if key in config:
            checker.require(f"config path resolves: {key}", _resolve_from(POST_EST_DIR, str(config[key])))

    parts = config.get("parts_cost_adjustment", {})
    checker.check("canonical parts adjustment mode", parts.get("mode") == "constant", str(parts))
    if checker.require("baseline cost-side pass-through coefficients exist", COST_SIDE_ROBUSTNESS_COEFFS):
        expected_pass_through = _baseline_cost_side_pass_through()
        configured_pass_through = parts.get("constant_pass_through")
        try:
            pass_through_ok = abs(float(configured_pass_through) - expected_pass_through) <= PASS_THROUGH_TOL
        except Exception:
            pass_through_ok = False
        checker.check(
            "canonical constant pass-through matches baseline cost-side coefficient",
            pass_through_ok,
            f"config={configured_pass_through}, cost_side={expected_pass_through:.6f}",
        )
    coeff_path = parts.get("coefficient_path")
    if coeff_path:
        checker.require("cost-side elasticity coefficient path resolves", _resolve_from(POST_EST_DIR, str(coeff_path)))


def validate_scenarios(checker: Checker, bundle: Path, metadata: dict[str, Any]) -> pd.DataFrame:
    run_cf_batch, default_scenario_specs = _load_run_modules()
    expected_specs = default_scenario_specs(
        parts_tariff=run_cf_batch.PARTS_TARIFF,
        vehicle_tariff=run_cf_batch.VEHICLE_TARIFF,
        country_tariffs=run_cf_batch.COUNTRY_TARIFFS,
    )

    scenario_path = bundle / "scenario_index.csv"
    checker.require("scenario index exists", scenario_path)
    scenario_index = pd.read_csv(scenario_path)
    expected_keys = set(expected_specs)
    actual_keys = set(scenario_index["scenario_key"].astype(str))
    checker.check("scenario keys match default scenario specs", actual_keys == expected_keys, f"actual={sorted(actual_keys)}")

    labels_ok = True
    slugs_ok = True
    for key, spec in expected_specs.items():
        row = scenario_index.loc[scenario_index["scenario_key"].astype(str).eq(key)]
        if row.empty:
            labels_ok = False
            slugs_ok = False
            continue
        label = str(row.iloc[0]["scenario_label"])
        slug = str(row.iloc[0]["label_slug"])
        labels_ok = labels_ok and label == spec.label
        slugs_ok = slugs_ok and slug == _slug(spec.label)
    checker.check("scenario labels match default specs", labels_ok, "labels are generated from ScenarioSpec")
    checker.check("scenario slugs match labels", slugs_ok, "slug(label) == label_slug")
    checker.check("metadata scenario label registry matches scenario index", set(metadata.get("scenarios", {}).values()) == set(scenario_index["scenario_label"]), str(metadata.get("scenarios", {})))
    return scenario_index


def validate_metadata(checker: Checker, bundle: Path, config: dict[str, Any]) -> dict[str, Any]:
    meta_path = bundle / "metadata.json"
    checker.require("canonical bundle metadata exists", meta_path)
    metadata = _load_json(meta_path)
    run_cf_batch, _ = _load_run_modules()

    checker.check("bundle is rebased to B0", metadata.get("reporting_baseline_label") == BASELINE_LABEL, str(metadata.get("reporting_baseline_label")))
    checker.check("baseline scenario is B0", metadata.get("baseline_scenario_label") == BASELINE_LABEL, str(metadata.get("baseline_scenario_label")))
    checker.check("metadata results file matches config", Path(str(metadata.get("results_file", ""))).resolve() == _config_results_path(config), str(metadata.get("results_file")))
    checker.check("metadata total market size matches code", _same_float(metadata.get("total_market_size"), run_cf_batch.TOTAL_MARKET_SIZE), str(metadata.get("total_market_size")))
    checker.check("metadata price scale matches code", _same_float(metadata.get("price_scale_usd_per_unit"), run_cf_batch.PRICE_SCALE_USD_PER_UNIT), str(metadata.get("price_scale_usd_per_unit")))
    checker.check("metadata vehicle tariff matches code", _same_float(metadata.get("vehicle_tariff"), run_cf_batch.VEHICLE_TARIFF), str(metadata.get("vehicle_tariff")))
    checker.check("metadata parts tariff matches code", _same_float(metadata.get("parts_tariff"), run_cf_batch.PARTS_TARIFF), str(metadata.get("parts_tariff")))
    checker.check("metadata country tariffs match code", _same_mapping(metadata.get("country_tariffs"), run_cf_batch.COUNTRY_TARIFFS), str(metadata.get("country_tariffs")))
    checker.check("code country tariffs are canonical", run_cf_batch.COUNTRY_TARIFFS == normalize_country_tariffs(run_cf_batch.COUNTRY_TARIFFS), str(run_cf_batch.COUNTRY_TARIFFS))
    checker.check("country-specific tariff keys use canonical Korea label", "Korea" in run_cf_batch.COUNTRY_TARIFFS and "South Korea" not in run_cf_batch.COUNTRY_TARIFFS, str(run_cf_batch.COUNTRY_TARIFFS))
    checker.check("metadata ownership mode matches config", metadata.get("ownership_mode") == config.get("ownership_mode"), str(metadata.get("ownership_mode")))

    parts = config.get("parts_cost_adjustment", {})
    checker.check("metadata parts pass-through mode matches config", metadata.get("parts_pass_through_mode") == parts.get("mode"), str(metadata.get("parts_pass_through_mode")))
    checker.check("metadata constant pass-through matches config", _same_float(metadata.get("parts_pass_through"), parts.get("constant_pass_through")), str(metadata.get("parts_pass_through")))
    return metadata


def _noncanonical_values(series: pd.Series) -> list[str]:
    values = {str(v).strip() for v in series.dropna().unique()}
    return sorted(values & NONCANONICAL_COUNTRY_LABELS)


def validate_country_consistency(checker: Checker, bundle: Path) -> None:
    parsing_summary = REPO_ROOT / "cost_side" / "outputs" / "country_parsing_summary.csv"
    if checker.require("cost-side country parsing summary exists", parsing_summary):
        summary = pd.read_csv(parsing_summary)
        bad = summary.loc[summary["canonical_country"].isna(), "raw_country"].astype(str).tolist()
        checker.check("cost-side country parser has no unmapped raw tokens", not bad, "; ".join(sorted(set(bad))))
        canonical_values = set(summary["canonical_country"].dropna().astype(str))
        checker.check("cost-side parser canonicalizes UK/GB to United Kingdom", "Great Britain" not in canonical_values and "United Kingdom" in canonical_values, str(sorted(canonical_values)))

    cost_panels = [
        REPO_ROOT / "cost_side" / "cost_side_panel_all.csv",
        REPO_ROOT / "cost_side" / "cost_side_panel.csv",
        REPO_ROOT / "cost_side" / "cost_side_panel_foreign.csv",
    ]
    for path in cost_panels:
        if not path.exists():
            continue
        df = pd.read_csv(
            path,
            usecols=lambda col: col
            in {
                "plant_country",
                "pcOth1_code1",
                "pcOth1_code2",
                "pcOth1_pct1",
                "rer_pcOth1_code1_n2015",
                "canonical_exposure_ok",
                "primary_source_country_consistent",
            },
        )
        for col in [c for c in df.columns if c in {"plant_country", "pcOth1_code1", "pcOth1_code2"}]:
            bad = _noncanonical_values(df[col])
            checker.check(f"{path.name} {col} has canonical country labels", not bad, ",".join(bad))
        if "rer_pcOth1_code1_n2015" in df.columns:
            missing_rer = int(pd.to_numeric(df["rer_pcOth1_code1_n2015"], errors="coerce").isna().sum())
            checker.check(f"{path.name} primary source-country RER is complete", missing_rer == 0, f"missing={missing_rer}")
        if "pcOth1_pct1" in df.columns:
            pct = pd.to_numeric(df["pcOth1_pct1"], errors="coerce")
            bad_pct = int((pct.isna() | pct.le(0) | pct.gt(1)).sum())
            checker.check(f"{path.name} primary import share is in (0, 1]", bad_pct == 0, f"bad={bad_pct}")

    all_cost_panel = REPO_ROOT / "cost_side" / "cost_side_panel_all.csv"
    domestic_cost_panel = REPO_ROOT / "cost_side" / "cost_side_panel.csv"
    if all_cost_panel.exists() and domestic_cost_panel.exists():
        all_df = pd.read_csv(
            all_cost_panel,
            usecols=[
                "product_ids",
                "market_year",
                "is_us_assembled",
                "canonical_exposure_ok",
                "primary_source_country_consistent",
            ],
        )
        dom_df = pd.read_csv(domestic_cost_panel, usecols=["product_ids", "market_year"])
        canonical = all_df.loc[
            all_df["is_us_assembled"].astype(str).str.lower().isin(["true", "1"])
            & all_df["canonical_exposure_ok"].astype(str).str.lower().isin(["true", "1"])
        ].copy()
        inconsistent = canonical.loc[
            ~canonical["primary_source_country_consistent"].astype(str).str.lower().isin(["true", "1"])
        ]
        checker.check(
            "canonical domestic panel has no raw primary-country conflicts",
            inconsistent.empty,
            f"bad={len(inconsistent)}",
        )
        canonical_keys = set(map(tuple, canonical[["product_ids", "market_year"]].astype(str).values))
        domestic_keys = set(map(tuple, dom_df[["product_ids", "market_year"]].astype(str).values))
        checker.check(
            "slim domestic cost-side panel matches canonical audit subset",
            domestic_keys == canonical_keys,
            f"domestic={len(domestic_keys)}, canonical={len(canonical_keys)}",
        )

    product_data_path = POST_EST_DIR / "data" / "raw" / "product_data_45W.csv"
    if product_data_path.exists():
        product_data = pd.read_csv(product_data_path, usecols=["market_year", "plant_country"])
        product_data["plant_country_canonical"] = normalize_country_series(product_data["plant_country"])
        raw_bad = _noncanonical_values(product_data["plant_country"])
        canonical_bad = _noncanonical_values(product_data["plant_country_canonical"])
        checker.check("raw product data may contain known aliases but normalizes cleanly", not canonical_bad, f"raw_aliases={raw_bad}; canonical_bad={canonical_bad}")
        countries_2024 = set(
            product_data.loc[
                pd.to_numeric(product_data["market_year"], errors="coerce").eq(2024),
                "plant_country_canonical",
            ].dropna().astype(str)
        )
        run_cf_batch, _ = _load_run_modules()
        tariff_keys = set(run_cf_batch.COUNTRY_TARIFFS or {})
        intended = {"United Kingdom", "Japan", "Korea"}
        checker.check("country-specific tariff keys are present in canonical 2024 product countries", intended.issubset(countries_2024 | tariff_keys) and intended.issubset(tariff_keys), f"countries={sorted(countries_2024)}, tariffs={sorted(tariff_keys)}")

    scenario_index_path = bundle / "scenario_index.csv"
    if scenario_index_path.exists():
        scenario_index = pd.read_csv(scenario_index_path)
        for _, row in scenario_index.iterrows():
            slug = str(row["label_slug"])
            product_path = bundle / f"{slug}__product_table.csv.gz"
            if not product_path.exists():
                continue
            product = pd.read_csv(product_path, usecols=lambda col: col == "plant_country")
            if "plant_country" in product.columns:
                bad = _noncanonical_values(product["plant_country"])
                checker.check(f"{slug} product table plant_country canonical", not bad, ",".join(bad))


def validate_bundle_files(checker: Checker, bundle: Path, scenario_index: pd.DataFrame) -> None:
    checker.require("summary table exists", bundle / "summary_tbl_all.csv.gz")
    checker.require("EV and tariff table exists", bundle / "ev_tariff_tbl.csv.gz")
    checker.require("profit graph values exist", bundle / "profit_changes_graph_values.csv")

    common_suffixes = [
        "product_table",
        "firm_table",
        "owner_table",
        "market_surplus_table",
        "overall_surplus",
        "origin_metrics",
        "state_cs",
        "state_units",
    ]
    figure_prefixes = ["origin_metrics", "profit_changes", "assembly_map", "cs_map"]
    figures_dir = bundle / "figures"
    checker.require("bundle figures directory exists", figures_dir)

    for _, row in scenario_index.iterrows():
        slug = str(row["label_slug"])
        label = str(row["scenario_label"])
        for suffix in common_suffixes:
            checker.require(f"{slug} {suffix}", bundle / f"{slug}__{suffix}.csv.gz")
        is_tariff_scenario = "vehicles_only_tariff" in slug or "parts_and_vehicles_tariff" in slug
        if is_tariff_scenario:
            checker.require(f"{slug} tariff cost table", bundle / f"{slug}__cf_costs_df.csv.gz")
        for prefix in figure_prefixes:
            checker.require(f"{slug} figure {prefix}", figures_dir / f"{prefix}_{slug}.png")
        checker.check(f"{slug} scenario label nonempty", bool(label), label)


def validate_numerical_invariants(checker: Checker, bundle: Path) -> None:
    summary = pd.read_csv(bundle / "summary_tbl_all.csv.gz", index_col=0)
    checker.check("summary contains B0 column", BASELINE_LABEL in summary.columns, ",".join(summary.columns))
    if BASELINE_LABEL not in summary.columns:
        return

    b0 = summary[BASELINE_LABEL]
    zero_rows = [
        row for row in summary.index
        if "Price" in row
        or "Producer Surplus" in row
        or row.startswith("CS ")
        or row.endswith("vehicles sold (millions)")
        or "US assembled" in row
        or "Net US impact" in row
    ]
    bad_rows: list[str] = []
    for row in zero_rows:
        try:
            value = _numeric_cell(b0.loc[row])
        except Exception:
            bad_rows.append(row)
            continue
        if not _abs_leq(value):
            bad_rows.append(f"{row}={value}")
    checker.check("B0 delta rows are zero", not bad_rows, "; ".join(bad_rows))

    ev_row_name = next((row for row in summary.index if row.startswith("EV share")), None)
    checker.check("summary has EV share row", ev_row_name is not None, str(list(summary.index)))

    ev_tbl = pd.read_csv(bundle / "ev_tariff_tbl.csv.gz")
    b0_ev = ev_tbl.loc[ev_tbl["Scenario"].eq(BASELINE_LABEL)]
    checker.check("EV table contains B0", not b0_ev.empty, "")
    if not b0_ev.empty:
        b0_ev = b0_ev.iloc[0]
        checker.check("B0 EV share change is zero", _abs_leq(float(b0_ev["Delta EV share (pp)"]) if "Delta EV share (pp)" in b0_ev else float(b0_ev["\u0394 EV share (pp)"])), str(dict(b0_ev)))
        checker.check("B0 units baseline equals counterfactual", _same_float(b0_ev["Units (baseline)"], b0_ev["Units (CF)"], tol=1e-4), str(dict(b0_ev)))
        checker.check("B0 tariff rates are zero", _same_float(b0_ev["parts_tariff"], 0.0) and _same_float(b0_ev["vehicle_tariff"], 0.0), str(dict(b0_ev)))
        if ev_row_name:
            checker.check("summary EV share matches EV table", _same_float(_numeric_cell(summary.loc[ev_row_name, BASELINE_LABEL]) / 100.0, b0_ev["EV share (CF)"], tol=1e-6), f"summary={summary.loc[ev_row_name, BASELINE_LABEL]}, ev_tbl={b0_ev['EV share (CF)']}")

    product_path = bundle / "no_tariff__no_subsidy__product_table.csv.gz"
    checker.require("B0 product table exists", product_path)
    if product_path.exists():
        pt = pd.read_csv(product_path)
        for col in ["dp", "dc", "ds", "dpi", "dmu", "dmargin_pct"]:
            if col in pt.columns:
                max_abs = float(pd.to_numeric(pt[col], errors="coerce").abs().max())
                checker.check(f"B0 product {col} is zero", max_abs <= TOL, f"max_abs={max_abs}")
        ev_units = float(pd.to_numeric(pt["s_cf"], errors="coerce").sum())
        metadata = _load_json(bundle / "metadata.json")
        expected_units = ev_units * float(metadata["total_market_size"])
        if not b0_ev.empty:
            checker.check("B0 product shares reproduce EV table units", _same_float(expected_units, b0_ev["Units (CF)"], tol=1e-3), f"from_product={expected_units}, table={b0_ev['Units (CF)']}")


def validate_cost_side_outputs(checker: Checker) -> None:
    required = [
        "cost_reg_table.tex",
        "cost_reg_elas_levels_table.tex",
        "cost_reg_elas_fd_table.tex",
        "cost_reg_elas_primary_spec_coeffs.csv",
        "panel_build_sample_counts.csv",
        "cost_reg_placebo_coefficients.csv",
        "cost_reg_robustness_coefficients.csv",
        "leave_one_country_out_coefficients.csv",
        "cost_reg_price_markup_decomp_coefficients.csv",
        "cost_reg_alt_exposure_coefficients.csv",
        "full_robustness_main_table.tex",
        "exchange_rate_2025_backfill_values.csv",
        "exchange_rate_2025_backfill_summary.md",
    ]
    for name in required:
        checker.require(f"cost-side output {name}", REPO_ROOT / "cost_side" / "outputs" / name)
    checker.require("full cost-side robustness report PDF", REPO_ROOT / "cost_side" / "docs" / "full_robustness_report.pdf")
    checker.require("cost-side robustness note PDF", REPO_ROOT / "cost_side" / "docs" / "cost_side_robustness_note.pdf")


def validate_paper_outputs(checker: Checker, bundle: Path, config: dict[str, Any]) -> None:
    manifest_path = REPO_ROOT / "paper" / "build" / "manifest.json"
    checker.require("paper build manifest exists", manifest_path)
    checker.require("reproduced paper PDF exists", REPO_ROOT / "paper" / "build" / "Auto_Tariffs.reproduced.pdf")
    checker.require("canonical paper PDF exists", REPO_ROOT / "paper" / "Auto_Tariffs.pdf")
    if not manifest_path.exists():
        return
    manifest = _load_json(manifest_path)
    checker.check("paper manifest validation succeeded", bool(manifest.get("validation_ok")), str(manifest.get("validation_ok")))
    checker.check("paper manifest rebased bundle is canonical", Path(str(manifest.get("rebased_bundle", ""))).resolve() == bundle, str(manifest.get("rebased_bundle")))
    checker.check("paper manifest results config is canonical", Path(str(manifest.get("results_config", ""))).resolve() == CANONICAL_CONFIG.resolve(), str(manifest.get("results_config")))
    checker.check("paper manifest results file matches config", Path(str(manifest.get("results_file", ""))).resolve() == _config_results_path(config), str(manifest.get("results_file")))


def write_report(checker: Checker, bundle: Path) -> None:
    out_dir = POST_EST_DIR / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "ok": checker.ok,
        "canonical_bundle": str(bundle),
        "checks": [asdict(result) for result in checker.results],
    }
    (out_dir / "downstream_consistency_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Downstream Consistency Report",
        "",
        f"- Overall status: {'PASS' if checker.ok else 'FAIL'}",
        f"- Canonical bundle: `{bundle}`",
        "",
        "| Status | Check | Detail |",
        "|---|---|---|",
    ]
    for result in checker.results:
        status = "PASS" if result.ok else "FAIL"
        detail = str(result.detail).replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {status} | {result.name} | {detail} |")
    (out_dir / "downstream_consistency_report.md").write_text("\n".join(lines) + "\n")


def run_checks(args: argparse.Namespace) -> tuple[Checker, Path]:
    checker = Checker()
    config = _load_json(CANONICAL_CONFIG)
    bundle = locate_canonical_bundle(config, Path(args.bundle) if args.bundle else None)

    validate_config(checker, config)
    metadata = validate_metadata(checker, bundle, config)
    scenario_index = validate_scenarios(checker, bundle, metadata)
    validate_country_consistency(checker, bundle)
    validate_bundle_files(checker, bundle, scenario_index)
    validate_numerical_invariants(checker, bundle)
    if not args.skip_cost_side:
        validate_cost_side_outputs(checker)
    if not args.skip_paper:
        validate_paper_outputs(checker, bundle, config)
    return checker, bundle


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate downstream replication consistency.")
    parser.add_argument("--bundle", help="Override the canonical rebased bundle path.")
    parser.add_argument("--print-canonical-bundle", action="store_true", help="Print the resolved canonical rebased bundle and exit.")
    parser.add_argument("--skip-cost-side", action="store_true", help="Skip cost-side generated-output checks.")
    parser.add_argument("--skip-paper", action="store_true", help="Skip paper generated-output checks.")
    parser.add_argument("--no-write-report", action="store_true", help="Do not write generated JSON/Markdown reports.")
    args = parser.parse_args()

    config = _load_json(CANONICAL_CONFIG)
    if args.print_canonical_bundle:
        print(locate_canonical_bundle(config, Path(args.bundle) if args.bundle else None))
        return 0

    checker, bundle = run_checks(args)
    for result in checker.results:
        prefix = "PASS" if result.ok else "FAIL"
        print(f"{prefix}: {result.name} - {result.detail}")
    if not args.no_write_report:
        write_report(checker, bundle)
        print(f"Wrote {POST_EST_DIR / 'outputs' / 'downstream_consistency_report.json'}")
        print(f"Wrote {POST_EST_DIR / 'outputs' / 'downstream_consistency_report.md'}")
    return 0 if checker.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
