from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from pypdf import PdfReader

REPO_ROOT = Path(__file__).resolve().parent.parent
MPLCONFIGDIR = REPO_ROOT / "paper" / "build" / "mplconfig"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR.resolve()))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paper_assets import (
    CANONICAL_DATE,
    build_counterfactual_tables,
    build_division_map,
    build_diversions_table,
    build_domestic_share_figure,
    build_foreign_parts_figure,
    build_markup_distribution_figure,
    build_price_coef_figure,
    build_profit_scatter_figures,
    build_sales_weighted_sources_figure,
    build_summary_by_type_table,
    build_vehicle_type_trends_figure,
    copy_counterfactual_figures,
    copy_generated_figure,
    copy_generated_table,
    discover_paths,
    ensure_generated_dirs,
    locate_rebased_bundle,
    normalize_pdf_text,
    parse_required_assets,
    sha256_file,
    strip_tex_comments,
    validate_generated_assets,
    write_references_bbl,
)


def run_command(cmd: list[str], cwd: Path, env: dict[str, str] | None = None) -> None:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    subprocess.run(cmd, cwd=cwd, env=merged_env, check=True)


def render_repo_outputs(paths) -> None:
    python = sys.executable

    run_command([python, "post_est/build_blp_est_table.py"], cwd=paths.repo_root)
    run_command([python, "post_est/build_micro_moments_table.py"], cwd=paths.repo_root)
    run_command([python, "post_est/build_ev_subsidy_tables.py"], cwd=paths.repo_root)
    run_command(["Rscript", "cost_side/cost_reg.R"], cwd=paths.repo_root)
    run_command([python, "post_est/run_cf_batch.py"], cwd=paths.repo_root)
    run_command([python, "post_est/rebase_saved_outputs_b0.py"], cwd=paths.repo_root)

    bundle = locate_rebased_bundle(paths)
    run_command(
        [python, "post_est/export_profit_change_graph_values.py"],
        cwd=paths.repo_root,
        env={"CF_SOURCE_BUNDLE": str(bundle.rebased_bundle.resolve())},
    )


def copy_generated_table_for_paper(src: Path, dst: Path, resize_to_textwidth: bool = False) -> None:
    text = src.read_text()
    if resize_to_textwidth:
        text = re.sub(
            r"(\\begin\{tabular\}\{[^}]+\})",
            r"\\resizebox{\\textwidth}{!}{%\n   \1",
            text,
            count=1,
        )
        text = text.replace(r"\end{tabular}", r"\end{tabular}%" + "\n   }", 1)
    dst.write_text(text)


def materialize_paper_assets(paths) -> dict[str, object]:
    bundle = locate_rebased_bundle(paths)

    build_price_coef_figure(paths)
    build_markup_distribution_figure(paths)
    build_vehicle_type_trends_figure(paths, out_names=["trends1.png", "trends.png"])
    build_domestic_share_figure(paths)
    build_sales_weighted_sources_figure(paths)
    build_foreign_parts_figure(paths)
    build_division_map(paths)
    build_diversions_table(paths)
    build_counterfactual_tables(paths, bundle)
    build_summary_by_type_table(paths)
    build_profit_scatter_figures(paths, bundle)
    write_references_bbl(paths)

    copy_generated_table(paths.repo_root / "post_est" / "outputs" / "blp_est.tex", paths.tables_dir / "blp_est.tex")
    copy_generated_table(
        paths.repo_root / "post_est" / "outputs" / "micro_moments.tex",
        paths.tables_dir / "micro_moments.tex",
    )
    copy_generated_table(
        paths.repo_root / "post_est" / "outputs" / "avg_ev_subsidy_by_producer_45W.tex",
        paths.tables_dir / "avg_ev_subsidy_by_producer_45W.tex",
    )
    copy_generated_table(
        paths.repo_root / "cost_side" / "outputs" / "cost_reg_table.tex",
        paths.tables_dir / "cost_reg_table.tex",
    )
    cost_side_robustness_tables = [
        "cost_reg_robustness_canonical_table.tex",
        "cost_reg_robustness_rer_main_table.tex",
        "cost_reg_placebo_levels_table.tex",
        "cost_reg_placebo_fd_table.tex",
        "cost_reg_timing_triplet_levels_table.tex",
        "cost_reg_timing_triplet_fd_table.tex",
        "cost_reg_price_markup_decomp_table.tex",
        "cost_reg_price_markup_forward_placebo_table.tex",
        "cost_reg_alt_exposure_table.tex",
        "leave_one_country_out_report_table.tex",
    ]
    wide_cost_side_tables = {
        "cost_reg_price_markup_decomp_table.tex",
        "cost_reg_price_markup_forward_placebo_table.tex",
        "cost_reg_alt_exposure_table.tex",
    }
    for table_name in cost_side_robustness_tables:
        copy_generated_table_for_paper(
            paths.repo_root / "cost_side" / "outputs" / table_name,
            paths.tables_dir / table_name,
            resize_to_textwidth=table_name in wide_cost_side_tables,
        )

    copied_counterfactual_figures = copy_counterfactual_figures(bundle, paths)

    return {
        "source_bundle": str(bundle.source_bundle.resolve()),
        "rebased_bundle": str(bundle.rebased_bundle.resolve()),
        "graph_values_csv": str(bundle.graph_values_csv.resolve()),
        "copied_counterfactual_figures": copied_counterfactual_figures,
    }


def compile_paper(paths) -> Path:
    latex_dir = paths.build_dir / "latex"
    if latex_dir.exists():
        shutil.rmtree(latex_dir)
    latex_dir.mkdir(parents=True, exist_ok=True)
    tex_cache_dir = paths.build_dir / "texlive-cache"
    tex_cache_dir.mkdir(parents=True, exist_ok=True)
    build_tex = paths.paper_dir / "_paper_build_source.tex"
    build_tex.write_text(strip_tex_comments(paths.canonical_tex.read_text()) + "\n")
    try:
        run_command(
            [
                "latexmk",
                "-pdf",
                "-g",
                "-interaction=nonstopmode",
                "-halt-on-error",
                "-file-line-error",
                "-outdir=build/latex",
                build_tex.name,
            ],
            cwd=paths.paper_dir,
            env={
                "LANG": "en_US.UTF-8",
                "LC_ALL": "en_US.UTF-8",
                "TEXMFVAR": str(tex_cache_dir.resolve()),
                "TEXMFCONFIG": str(tex_cache_dir.resolve()),
                "VARTEXFONTS": str(tex_cache_dir.resolve()),
            },
        )
    finally:
        if build_tex.exists():
            build_tex.unlink()

    compiled_pdf = latex_dir / f"{build_tex.stem}.pdf"
    if not compiled_pdf.exists():
        raise FileNotFoundError(f"Expected compiled PDF not found: {compiled_pdf}")
    reproduced_pdf = paths.build_dir / "Auto_Tariffs.reproduced.pdf"
    shutil.copy2(compiled_pdf, reproduced_pdf)
    return reproduced_pdf


def _read_pdf_text(pdf_path: Path) -> tuple[int, str]:
    reader = PdfReader(str(pdf_path))
    pages: list[str] = []
    for page in reader.pages:
        text = normalize_pdf_text(page.extract_text() or "")
        text = re.sub(r"\s\d{1,3}$", "", text).strip()
        pages.append(text)
    return len(reader.pages), "\n".join(pages)


def _diff_excerpt(left: str, right: str, context: int = 2) -> str:
    left_lines = left.split(" ")
    right_lines = right.split(" ")
    diff = list(difflib.unified_diff(left_lines, right_lines, lineterm="", n=context))
    if not diff:
        return ""
    return "\n".join(diff[:40])


def _write_pdf_comparison_artifacts(paths, canonical_text: str, rebuilt_text: str, diff_excerpt: str) -> dict[str, str]:
    canonical_text_path = paths.build_dir / "canonical_pdf.normalized.txt"
    rebuilt_text_path = paths.build_dir / "reproduced_pdf.normalized.txt"
    diff_path = paths.build_dir / "canonical_pdf.diff"

    canonical_text_path.write_text(canonical_text + "\n")
    rebuilt_text_path.write_text(rebuilt_text + "\n")
    diff_path.write_text(diff_excerpt + ("\n" if diff_excerpt else ""))

    return {
        "canonical_pdf_normalized_text": str(canonical_text_path.resolve()),
        "reproduced_pdf_normalized_text": str(rebuilt_text_path.resolve()),
        "canonical_pdf_diff": str(diff_path.resolve()),
    }


def validate_paper(paths, reproduced_pdf: Path | None) -> dict[str, object]:
    results = validate_generated_assets(paths)

    if reproduced_pdf is None or not reproduced_pdf.exists():
        results.update(
            {
                "compiled_pdf_exists": False,
                "pdf_page_count_match": False,
                "pdf_text_match": False,
                "pdf_diff_excerpt": "Compiled PDF missing.",
                "canonical_pdf_match_ok": False,
            }
        )
        return results

    results["compiled_pdf_exists"] = True
    canonical_pages, canonical_text = _read_pdf_text(paths.canonical_pdf)
    rebuilt_pages, rebuilt_text = _read_pdf_text(reproduced_pdf)
    results["canonical_pdf_pages"] = canonical_pages
    results["rebuilt_pdf_pages"] = rebuilt_pages
    results["pdf_page_count_match"] = canonical_pages == rebuilt_pages
    results["pdf_text_match"] = canonical_text == rebuilt_text
    results["pdf_diff_excerpt"] = "" if canonical_text == rebuilt_text else _diff_excerpt(canonical_text, rebuilt_text)
    results["canonical_pdf_match_ok"] = results["pdf_page_count_match"] and results["pdf_text_match"]
    results.update(_write_pdf_comparison_artifacts(paths, canonical_text, rebuilt_text, results["pdf_diff_excerpt"]))
    results["validation_ok"] = results["asset_check_ok"] and results["compiled_pdf_exists"]
    return results


def build_manifest(paths, asset_info: dict[str, object], reproduced_pdf: Path | None, validation: dict[str, object]) -> dict[str, object]:
    results_file = json.loads(paths.results_config.read_text())["results_file"]
    results_path = Path(results_file)
    if not results_path.is_absolute():
        results_path = (paths.results_config.parent / results_path).resolve()

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "canonical_date": CANONICAL_DATE,
        "canonical_tex": str(paths.canonical_tex.resolve()),
        "canonical_tex_sha256": sha256_file(paths.canonical_tex),
        "canonical_pdf": str(paths.canonical_pdf.resolve()),
        "canonical_pdf_sha256": sha256_file(paths.canonical_pdf),
        "results_config": str(paths.results_config.resolve()),
        "results_config_sha256": sha256_file(paths.results_config),
        "results_file": str(results_path),
        "results_file_sha256": sha256_file(results_path),
        "required_assets": parse_required_assets(paths.canonical_tex),
        "generated_tables_dir": str(paths.tables_dir.resolve()),
        "generated_graphs_dir": str(paths.graphs_dir.resolve()),
        "build_dir": str(paths.build_dir.resolve()),
        "reproduced_pdf": str(reproduced_pdf.resolve()) if reproduced_pdf is not None and reproduced_pdf.exists() else None,
        "reproduced_pdf_sha256": sha256_file(reproduced_pdf) if reproduced_pdf is not None and reproduced_pdf.exists() else None,
    }
    manifest.update(asset_info)
    manifest.update(validation)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and validate the canonical paper.")
    parser.add_argument("--skip-render", action="store_true", help="Skip rerunning upstream generation scripts.")
    parser.add_argument("--skip-compile", action="store_true", help="Skip LaTeX compilation.")
    parser.add_argument("--skip-validation", action="store_true", help="Skip final validation checks.")
    parser.add_argument(
        "--strict-canonical-pdf",
        action="store_true",
        help="Fail if the rebuilt PDF does not exactly match paper/Auto_Tariffs.pdf after normalization.",
    )
    args = parser.parse_args()

    paths = discover_paths()
    ensure_generated_dirs(paths)

    if not args.skip_render:
        render_repo_outputs(paths)
    asset_info = materialize_paper_assets(paths)

    reproduced_pdf = None if args.skip_compile else compile_paper(paths)
    validation = {"validation_ok": True}
    if not args.skip_validation:
        validation = validate_paper(paths, reproduced_pdf)

    manifest = build_manifest(paths, asset_info, reproduced_pdf, validation)
    manifest_path = paths.build_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    if not args.skip_validation and not validation.get("validation_ok", False):
        return 1
    if args.strict_canonical_pdf and not validation.get("canonical_pdf_match_ok", False):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
