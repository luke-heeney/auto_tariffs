from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_cf_state_tables import build_plant_location_table, build_state_cs_table
from helpers.counterfactual_helpers import origin_percent_metrics, plot_origin_percent_metrics_bw
from helpers.counterfactual_reporting import (
    build_profit_change_artifacts,
    build_state_cs_map_figure,
    build_state_map_figure,
)
from helpers.figure_export import save_plotly_figure as _save_plotly_figure
from run_cf_batch import STATE_ABBR, STATE_CENTROIDS, US_FIRMS


BASELINE_LABEL = "no tariff (no subsidy)"
TARGET_LABEL_FOR_SCATTER = "parts and vehicles tariff (with subsidy)"

CS_ROWS = {
    "CS Δ total (billion USD)",
    "CS Δ Q1 (billion USD)",
    "CS Δ Q2 (billion USD)",
    "CS Δ Q3 (billion USD)",
    "CS Δ Q4 (billion USD)",
    "CS Δ Q5 (billion USD)",
}
DELTA_ROWS = {
    "Sales-weighted Δ Price (%)",
    "US Producer Surplus (Δ, billion USD)",
    "Δ vehicles sold (millions)",
    "Δ US assembled (millions)",
    "Net US impact (billion USD)",
}
CS_PATTERN = re.compile(r"\s*([+-]?\d+(?:\.\d+)?)\s*\(([-+]?\d+(?:\.\d+)?)%\)\s*")


def _latest_complete_saved_output_dir() -> Path:
    base = Path("post_est/saved_outputs")
    candidates: list[Path] = []
    for d in base.iterdir():
        if not d.is_dir():
            continue
        if "_rebased_b0" in d.name:
            continue
        if (d / "metadata.json").exists() and (d / "summary_tbl_all.csv.gz").exists() and (d / "scenario_index.csv").exists():
            candidates.append(d)
    if not candidates:
        raise FileNotFoundError("No complete saved output directories found.")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _parse_cs(v: object) -> tuple[float, float]:
    s = str(v).strip()
    m = CS_PATTERN.fullmatch(s)
    if m is None:
        raise ValueError(f"Could not parse CS cell: {v!r}")
    return float(m.group(1)), float(m.group(2))


def _parse_float(v: object) -> float:
    if isinstance(v, (int, float, np.floating)):
        return float(v)
    s = str(v).strip()
    if " (" in s:
        s = s.split(" (", 1)[0].strip()
    return float(s)


def _fmt_cs(delta: float, pct: float) -> str:
    if np.isfinite(pct):
        return f"{delta:.3f} ({pct:.1f}%)"
    return f"{delta:.3f} (nan)"


def _clean_zero(x: float, tol: float = 5e-10) -> float:
    return 0.0 if np.isfinite(x) and abs(x) <= tol else x


def _rebase_summary_tbl(summary: pd.DataFrame, baseline_label: str) -> pd.DataFrame:
    if baseline_label not in summary.columns:
        raise KeyError(f"Baseline scenario {baseline_label!r} missing from summary.")
    out = summary.copy()
    for row_name in summary.index:
        b0_raw = summary.loc[row_name, baseline_label]
        if row_name in CS_ROWS:
            d_b0, pct_b0 = _parse_cs(b0_raw)
            if pct_b0 == 0:
                cs_b0_level = np.nan
            else:
                cs_old_baseline = d_b0 / (pct_b0 / 100.0)
                cs_b0_level = cs_old_baseline + d_b0
            for col in summary.columns:
                d_old, _ = _parse_cs(summary.loc[row_name, col])
                d_new = _clean_zero(d_old - d_b0)
                pct_new = np.nan if (not np.isfinite(cs_b0_level) or cs_b0_level == 0) else 100.0 * d_new / cs_b0_level
                out.loc[row_name, col] = _fmt_cs(d_new, pct_new)
        elif row_name in DELTA_ROWS:
            b0_val = _parse_float(b0_raw)
            for col in summary.columns:
                val = _parse_float(summary.loc[row_name, col])
                out.loc[row_name, col] = _clean_zero(val - b0_val)
    return out


def _save_matplotlib_figure(fig, path: Path) -> None:
    if fig is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _assert_merge_ok(df: pd.DataFrame, key_cols: list[str], name: str) -> None:
    if df[key_cols].duplicated().any():
        raise ValueError(f"Duplicate keys found in {name} for {key_cols}")


def _rebase_product_table(pt: pd.DataFrame, b0_pt: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["market_ids", "product_ids"]
    pt = pt.copy()
    b0_pt = b0_pt.copy()
    _assert_merge_ok(pt, key_cols, "product_table")
    _assert_merge_ok(b0_pt, key_cols, "baseline product_table")

    base_cols = [
        "p_cf", "c_cf", "s_cf", "mu_cf", "margin_cf_pct", "pi_cf",
        "subsidy_cf", "p_cf_net",
    ]
    keep_base_cols = [c for c in base_cols if c in b0_pt.columns]
    b0_small = b0_pt[key_cols + keep_base_cols].rename(columns={c: f"b0_{c}" for c in keep_base_cols})

    out = pt.merge(b0_small, on=key_cols, how="left", validate="one_to_one")

    # Rebased baseline levels are the B0 scenario's counterfactual levels.
    out["p0"] = out["b0_p_cf"]
    out["c0"] = out["b0_c_cf"]
    out["s0"] = out["b0_s_cf"]
    if "b0_mu_cf" in out.columns:
        out["mu0"] = out["b0_mu_cf"]
    if "b0_margin_cf_pct" in out.columns:
        out["margin0_pct"] = out["b0_margin_cf_pct"]
    if "b0_pi_cf" in out.columns:
        out["pi0"] = out["b0_pi_cf"]
    if "subsidy0" in out.columns and "b0_subsidy_cf" in out.columns:
        out["subsidy0"] = out["b0_subsidy_cf"]
    if "p0_net" in out.columns and "b0_p_cf_net" in out.columns:
        out["p0_net"] = out["b0_p_cf_net"]

    # Recompute deltas and percentage changes.
    out["dp"] = out["p_cf"] - out["p0"]
    out["dp_pct"] = 100.0 * np.where(out["p0"] != 0, out["dp"] / out["p0"], np.nan)
    out["dc"] = out["c_cf"] - out["c0"]
    out["dc_pct"] = 100.0 * np.where(out["c0"] != 0, out["dc"] / out["c0"], np.nan)
    out["ds"] = out["s_cf"] - out["s0"]
    out["ds_pct"] = 100.0 * np.where(out["s0"] != 0, out["ds"] / out["s0"], np.nan)
    if "mu_cf" in out.columns and "mu0" in out.columns:
        out["dmu"] = out["mu_cf"] - out["mu0"]
    if "margin_cf_pct" in out.columns and "margin0_pct" in out.columns:
        out["dmargin_pct"] = out["margin_cf_pct"] - out["margin0_pct"]
    if "pi_cf" in out.columns and "pi0" in out.columns:
        out["dpi"] = out["pi_cf"] - out["pi0"]

    drop_cols = [c for c in out.columns if c.startswith("b0_")]
    return out.drop(columns=drop_cols)


def _rebase_group_table(tbl: pd.DataFrame, b0_tbl: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if tbl is None or b0_tbl is None:
        return tbl
    tbl = tbl.copy()
    b0_tbl = b0_tbl.copy()
    _assert_merge_ok(tbl, [group_col], f"{group_col} table")
    _assert_merge_ok(b0_tbl, [group_col], f"baseline {group_col} table")

    base_map = {
        "share0_total": "share_cf_total",
        "p0_sw": "p_cf_sw",
        "c0_sw": "c_cf_sw",
        "mu0_sw": "mu_cf_sw",
        "pi0_percap_total": "pi_cf_percap_total",
        "pi0_millions_usd": "pi_cf_millions_usd",
    }
    src_cols = [v for v in base_map.values() if v in b0_tbl.columns]
    b0_small = b0_tbl[[group_col] + src_cols].rename(columns={v: f"b0_{v}" for v in src_cols})
    out = tbl.merge(b0_small, on=group_col, how="left", validate="one_to_one")

    for dst, src in base_map.items():
        bcol = f"b0_{src}"
        if dst in out.columns and bcol in out.columns:
            out[dst] = out[bcol]

    for col in ["dp_sw", "dc_sw", "dmu_sw", "dpi_percap_total", "dpi_millions_usd"]:
        if col in out.columns:
            if col == "dp_sw":
                out[col] = out["p_cf_sw"] - out["p0_sw"]
            elif col == "dc_sw":
                out[col] = out["c_cf_sw"] - out["c0_sw"]
            elif col == "dmu_sw":
                out[col] = out["mu_cf_sw"] - out["mu0_sw"]
            elif col == "dpi_percap_total":
                out[col] = out["pi_cf_percap_total"] - out["pi0_percap_total"]
            elif col == "dpi_millions_usd":
                out[col] = out["pi_cf_millions_usd"] - out["pi0_millions_usd"]

    drop_cols = [c for c in out.columns if c.startswith("b0_")]
    # Preserve display order (sorted by profit change like original outputs).
    if "dpi_millions_usd" in out.columns:
        out = out.sort_values("dpi_millions_usd", ascending=False).reset_index(drop=True)
    return out.drop(columns=drop_cols)


def _rebase_market_surplus_table(tbl: pd.DataFrame, b0_tbl: pd.DataFrame) -> pd.DataFrame:
    key = ["market_ids"]
    tbl = tbl.copy()
    b0_tbl = b0_tbl.copy()
    _assert_merge_ok(tbl, key, "market_surplus_table")
    _assert_merge_ok(b0_tbl, key, "baseline market_surplus_table")
    b0_small = b0_tbl[key + ["CS_cf"]].rename(columns={"CS_cf": "b0_CS_cf"})
    out = tbl.merge(b0_small, on=key, how="left", validate="one_to_one")
    out["CS0"] = out["b0_CS_cf"]
    out["dCS"] = out["CS_cf"] - out["CS0"]
    if "CS0_millions_usd" in out.columns and "CS_cf_millions_usd" in out.columns:
        # Derive a factor per-row from existing values when possible.
        factor = np.where(out["CS_cf"] != 0, out["CS_cf_millions_usd"] / out["CS_cf"], np.nan)
        factor = pd.Series(factor).replace([np.inf, -np.inf], np.nan)
        fill_factor = float(factor.dropna().iloc[0]) if factor.notna().any() else np.nan
        factor = factor.fillna(fill_factor)
        out["CS0_millions_usd"] = out["CS0"] * factor
        out["dCS_millions_usd"] = out["CS_cf_millions_usd"] - out["CS0_millions_usd"]
    return out.drop(columns=["b0_CS_cf"])


def _rebase_state_units(tbl: pd.DataFrame, b0_tbl: pd.DataFrame) -> pd.DataFrame:
    key = ["plant_location"]
    tbl = tbl.copy()
    b0_tbl = b0_tbl.copy()
    _assert_merge_ok(tbl, key, "state_units")
    _assert_merge_ok(b0_tbl, key, "baseline state_units")
    b0_small = b0_tbl[key + ["units_cf"]].rename(columns={"units_cf": "b0_units_cf"})
    out = tbl.merge(b0_small, on=key, how="left", validate="one_to_one")
    out["units_base"] = out["b0_units_cf"]
    out["delta_units"] = out["units_cf"] - out["units_base"]
    out["pct_change"] = np.where(out["units_base"] != 0, 100.0 * out["delta_units"] / out["units_base"], np.nan)
    return out.drop(columns=["b0_units_cf"])


def _rebase_state_cs(tbl: pd.DataFrame, b0_tbl: pd.DataFrame) -> pd.DataFrame:
    key = ["state"]
    tbl = tbl.copy()
    b0_tbl = b0_tbl.copy()
    _assert_merge_ok(tbl, key, "state_cs")
    _assert_merge_ok(b0_tbl, key, "baseline state_cs")
    b0_small = b0_tbl[key + ["CS_cf"]].rename(columns={"CS_cf": "b0_CS_cf"})
    out = tbl.merge(b0_small, on=key, how="left", validate="one_to_one")
    out["CS0"] = out["b0_CS_cf"]
    out["dCS"] = out["CS_cf"] - out["CS0"]
    out["pct_change_vs_baseline"] = np.where(out["CS0"] != 0, 100.0 * out["dCS"] / out["CS0"], np.nan)
    return out.drop(columns=["b0_CS_cf"])


def _rebase_overall_surplus(tbl: pd.DataFrame, firm_tbl: pd.DataFrame, market_tbl: pd.DataFrame) -> pd.DataFrame:
    tbl = tbl.copy()
    if tbl.empty:
        return tbl
    if "total_firm_surplus_change_millions_usd" in tbl.columns:
        tbl.loc[:, "total_firm_surplus_change_millions_usd"] = float(firm_tbl["dpi_millions_usd"].sum())
    if "total_consumer_surplus_change_millions_usd" in tbl.columns and "dCS_millions_usd" in market_tbl.columns:
        tbl.loc[:, "total_consumer_surplus_change_millions_usd"] = float(market_tbl["dCS_millions_usd"].sum())
    return tbl


def _rebase_ev_tariff_tbl(ev_tbl: pd.DataFrame, baseline_label: str) -> pd.DataFrame:
    out = ev_tbl.copy()
    b_row = out.loc[out["Scenario"] == baseline_label]
    if b_row.empty:
        raise KeyError(f"Baseline scenario {baseline_label!r} not found in ev_tariff_tbl")
    b_row = b_row.iloc[0]
    b_ev = float(b_row["EV share (CF)"])
    b_units = float(b_row["Units (CF)"])
    out["EV share (baseline)"] = b_ev
    out["Units (baseline)"] = b_units
    out["Δ EV share (pp)"] = 100.0 * (pd.to_numeric(out["EV share (CF)"], errors="coerce") - b_ev)
    return out


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _rebase_profit_vs_import_share(
    firm_tbl: pd.DataFrame,
    product_tbl: pd.DataFrame,
    cf_costs_df: pd.DataFrame | None,
    out_path: Path,
) -> None:
    if cf_costs_df is None:
        return
    pt = product_tbl.copy()
    if "plant_country" not in pt.columns:
        return
    m = pt[pt["plant_country"].astype(str).str.strip().eq("United States")].copy()
    if m.empty:
        return
    if "pcUSCA_pct" not in cf_costs_df.columns:
        return
    pc_map = cf_costs_df[["product_ids", "pcUSCA_pct"]].drop_duplicates("product_ids")
    m = m.merge(pc_map, on="product_ids", how="left")
    m["import_share"] = 1.0 - pd.to_numeric(m["pcUSCA_pct"], errors="coerce")
    w = pd.to_numeric(m["s0"], errors="coerce").to_numpy(dtype=float)
    m["w"] = w
    firm_import = (
        m.groupby("firm_ids", dropna=False)
        .apply(lambda d: np.nan if d["w"].sum() == 0 else np.nansum(d["import_share"] * d["w"]) / np.nansum(d["w"]))
        .reset_index(name="import_share")
    )
    f = firm_tbl.copy()
    f["pct_change_profit"] = 100.0 * pd.to_numeric(f["dpi_millions_usd"], errors="coerce") / pd.to_numeric(
        f["pi0_millions_usd"], errors="coerce"
    )
    plot_df = f.merge(firm_import, on="firm_ids", how="inner")
    plot_df = plot_df[np.isfinite(plot_df["import_share"])].copy()
    plot_df["import_share_pct"] = 100.0 * plot_df["import_share"]
    plot_df["is_us"] = plot_df["firm_ids"].astype(str).str.lower().isin({x.lower() for x in US_FIRMS})
    plot_df = plot_df[plot_df["is_us"]].copy()
    plot_df = plot_df[~plot_df["firm_ids"].astype(str).str.lower().eq("rivian")].copy()
    if plot_df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(plot_df["import_share_pct"], plot_df["pct_change_profit"], color="0.2")
    for _, r in plot_df.iterrows():
        ax.annotate(str(r["firm_ids"]), (r["import_share_pct"], r["pct_change_profit"]), textcoords="offset points", xytext=(4, 2), ha="left", fontsize=8)
    ax.set_xlabel("Parts import share for US-assembled vehicles (%)")
    ax.set_ylabel("Firm profit change (%)")
    plt.tight_layout()
    _save_matplotlib_figure(fig, out_path)


def main() -> None:
    src_override = os.environ.get("CF_SOURCE_BUNDLE")
    if src_override:
        src_dir = Path(src_override)
    else:
        src_dir = _latest_complete_saved_output_dir()
    dst_dir = src_dir.parent / f"{src_dir.name}_rebased_b0"
    dst_dir.mkdir(parents=True, exist_ok=True)
    (dst_dir / "figures").mkdir(parents=True, exist_ok=True)

    print(f"Source bundle: {src_dir}")
    print(f"Target rebased bundle: {dst_dir}")

    scenario_index = pd.read_csv(src_dir / "scenario_index.csv")
    scenario_rows = scenario_index.to_dict("records")

    # Identify baseline scenario files.
    b0_row = next((r for r in scenario_rows if r["scenario_label"] == BASELINE_LABEL), None)
    if b0_row is None:
        raise KeyError(f"Baseline scenario {BASELINE_LABEL!r} not found in scenario_index.")
    b0_slug = b0_row["label_slug"]

    b0_product = pd.read_csv(src_dir / f"{b0_slug}__product_table.csv.gz")
    b0_firm = pd.read_csv(src_dir / f"{b0_slug}__firm_table.csv.gz")
    b0_market = pd.read_csv(src_dir / f"{b0_slug}__market_surplus_table.csv.gz")
    b0_owner = pd.read_csv(src_dir / f"{b0_slug}__owner_table.csv.gz") if (src_dir / f"{b0_slug}__owner_table.csv.gz").exists() else None
    b0_state_units = pd.read_csv(src_dir / f"{b0_slug}__state_units.csv.gz") if (src_dir / f"{b0_slug}__state_units.csv.gz").exists() else None
    b0_state_cs = pd.read_csv(src_dir / f"{b0_slug}__state_cs.csv.gz") if (src_dir / f"{b0_slug}__state_cs.csv.gz").exists() else None

    # Copy index now; same scenario list/labels.
    scenario_index.to_csv(dst_dir / "scenario_index.csv", index=False)

    target_scatter_inputs: tuple[pd.DataFrame, pd.DataFrame | None] | None = None

    for row in scenario_rows:
        label = row["scenario_label"]
        slug = row["label_slug"]

        pt = pd.read_csv(src_dir / f"{slug}__product_table.csv.gz")
        ft = pd.read_csv(src_dir / f"{slug}__firm_table.csv.gz")
        mt = pd.read_csv(src_dir / f"{slug}__market_surplus_table.csv.gz")
        ov = pd.read_csv(src_dir / f"{slug}__overall_surplus.csv.gz")
        ot = pd.read_csv(src_dir / f"{slug}__owner_table.csv.gz") if (src_dir / f"{slug}__owner_table.csv.gz").exists() else None
        su = pd.read_csv(src_dir / f"{slug}__state_units.csv.gz") if (src_dir / f"{slug}__state_units.csv.gz").exists() else None
        scs = pd.read_csv(src_dir / f"{slug}__state_cs.csv.gz") if (src_dir / f"{slug}__state_cs.csv.gz").exists() else None

        pt_r = _rebase_product_table(pt, b0_product)
        ft_r = _rebase_group_table(ft, b0_firm, "firm_ids")
        mt_r = _rebase_market_surplus_table(mt, b0_market)
        ov_r = _rebase_overall_surplus(ov, ft_r, mt_r)
        ot_r = _rebase_group_table(ot, b0_owner, "owner_ids") if ot is not None and b0_owner is not None else ot
        su_r = _rebase_state_units(su, b0_state_units) if su is not None and b0_state_units is not None else su
        scs_r = _rebase_state_cs(scs, b0_state_cs) if scs is not None and b0_state_cs is not None else scs

        pt_r.to_csv(dst_dir / f"{slug}__product_table.csv.gz", index=False, compression="gzip")
        ft_r.to_csv(dst_dir / f"{slug}__firm_table.csv.gz", index=False, compression="gzip")
        mt_r.to_csv(dst_dir / f"{slug}__market_surplus_table.csv.gz", index=False, compression="gzip")
        ov_r.to_csv(dst_dir / f"{slug}__overall_surplus.csv.gz", index=False, compression="gzip")
        if ot_r is not None:
            ot_r.to_csv(dst_dir / f"{slug}__owner_table.csv.gz", index=False, compression="gzip")
        if su_r is not None:
            su_r.to_csv(dst_dir / f"{slug}__state_units.csv.gz", index=False, compression="gzip")
        if scs_r is not None:
            scs_r.to_csv(dst_dir / f"{slug}__state_cs.csv.gz", index=False, compression="gzip")

        # Copy cost tables unchanged (level outputs, not baseline-relative).
        _copy_if_exists(src_dir / f"{slug}__cf_costs_df.csv.gz", dst_dir / f"{slug}__cf_costs_df.csv.gz")

        # Recompute origin metrics from rebased product table.
        if "plant_country" in pt_r.columns:
            origin_tbl = origin_percent_metrics(pt_r).reset_index()
            if "results_file" in pt_r.columns:
                origin_tbl["results_file"] = pt_r["results_file"].iloc[0]
            origin_tbl["scenario_label"] = label
            origin_tbl.to_csv(dst_dir / f"{slug}__origin_metrics.csv.gz", index=False, compression="gzip")

        # Figures
        fig_dir = dst_dir / "figures"
        profit_art = build_profit_change_artifacts(ft_r, scenario_label=label, us_firms=US_FIRMS)
        _save_matplotlib_figure(profit_art.get("figure"), fig_dir / f"profit_changes_{slug}.png")

        if "plant_country" in pt_r.columns:
            origin_tbl_idx = origin_percent_metrics(pt_r)
            fig = plot_origin_percent_metrics_bw(origin_tbl_idx, title=None, show=False)
            _save_matplotlib_figure(fig, fig_dir / f"origin_metrics_{slug}.png")

        if su_r is not None and not su_r.empty:
            fig = build_state_map_figure(
                su_r,
                scenario_label=label,
                state_abbr=STATE_ABBR,
                state_centroids=STATE_CENTROIDS,
            )
            _save_plotly_figure(fig, fig_dir / f"assembly_map_{slug}")

        if scs_r is not None and not scs_r.empty:
            fig = build_state_cs_map_figure(
                scs_r,
                scenario_label=label,
                state_abbr=STATE_ABBR,
                state_centroids=STATE_CENTROIDS,
            )
            _save_plotly_figure(fig, fig_dir / f"cs_map_{slug}")

        if label == TARGET_LABEL_FOR_SCATTER:
            cf_costs = pd.read_csv(src_dir / f"{slug}__cf_costs_df.csv.gz") if (src_dir / f"{slug}__cf_costs_df.csv.gz").exists() else None
            target_scatter_inputs = (ft_r, cf_costs)
            _rebase_profit_vs_import_share(
                firm_tbl=ft_r,
                product_tbl=pt_r,
                cf_costs_df=cf_costs,
                out_path=fig_dir / f"profit_change_vs_import_share_{slug}.png",
            )

    # Rebase summary / EV-tariff tables from source.
    summary_src = pd.read_csv(src_dir / "summary_tbl_all.csv.gz", index_col=0)
    summary_rebased = _rebase_summary_tbl(summary_src, BASELINE_LABEL)
    summary_rebased.to_csv(dst_dir / "summary_tbl_all.csv.gz", compression="gzip")

    ev_tbl_src = pd.read_csv(src_dir / "ev_tariff_tbl.csv.gz")
    ev_tbl_rebased = _rebase_ev_tariff_tbl(ev_tbl_src, BASELINE_LABEL)
    ev_tbl_rebased.to_csv(dst_dir / "ev_tariff_tbl.csv.gz", index=False, compression="gzip")

    # Preserve source rebased machine-readable summary if it exists.
    _copy_if_exists(src_dir / "summary_tbl_all_rebased_b0.csv.gz", dst_dir / "summary_tbl_all_rebased_b0.csv.gz")

    meta = json.loads((src_dir / "metadata.json").read_text())
    meta["source_saved_output_dir"] = str(src_dir.resolve())
    meta["reporting_baseline_label"] = BASELINE_LABEL
    meta["reporting_baseline_type"] = "scenario_cf_level"
    meta["reporting_baseline_slug"] = b0_slug
    meta["baseline_subsidy_spend_billion_usd"] = 0.0
    (dst_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    # Regenerate the state tables from the rebased bundle into post_est/outputs.
    out_dir = Path("post_est/outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plant_location_changes.tex").write_text(build_plant_location_table(dst_dir))
    (out_dir / "cs_by_state.tex").write_text(build_state_cs_table(dst_dir))

    print(f"Saved rebased bundle: {dst_dir}")
    print(f"Saved rebased state tables: {out_dir / 'plant_location_changes.tex'} and {out_dir / 'cs_by_state.tex'}")


if __name__ == "__main__":
    main()
