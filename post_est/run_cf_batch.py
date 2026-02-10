from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from helpers.counterfactual_costs_prep import prepare_costs_df2_for_year
from helpers.counterfactual_reporting import default_scenario_specs, run_scenario_outputs
from helpers.consumer_surplus import cs_change_by_income_bins
from helpers.ev_tariff_metrics import build_ev_and_tariff_table
from helpers.counterfactual_helpers import (
    origin_percent_metrics,
    plot_origin_percent_metrics_bw,
    build_costs_vector_from_vehicle_costs,
)
from helpers.ownership import load_owner_map, load_pricer_map
from helpers.counterfactual_reporting import (
    build_profit_change_artifacts,
    build_state_units_table,
    build_state_map_figure,
    build_state_cs_table,
    build_state_cs_map_figure,
)

# ---- config (mirrors run_cf.ipynb) ----
PRICE_X2_INDEX = 1
PRICE_BETA_INDEX = 0
CS_GAMMA = 0.0
INCOME_DEMO_INDEX = 0
INCOME_TRANSFORM = "log_10k"
CS_MARKET_ID = 2024

COUNTRY_TARIFFS = {
    "United Kingdom": 0.10,
    "Japan": 0.15,
    "South Korea": 0.15,
}

PARTS_TARIFF = 0.25
VEHICLE_TARIFF = 0.25

TOTAL_MARKET_SIZE = 132_216_000 / 6
PRICE_SCALE_USD_PER_UNIT = 100_000.0

US_FIRMS = [
    "ford", "chevrolet", "gmc", "buick", "cadillac", "chrysler",
    "ram", "jeep", "dodge", "tesla", "rivian", "lucid", "lincoln", "lucidmotors",
]

STATE_ABBR = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID",
    "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT",
    "Vermont": "VT", "Virginia": "VA", "Washington": "WA", "West Virginia": "WV",
    "Wisconsin": "WI", "Wyoming": "WY",
}

STATE_CENTROIDS = {
    "AL": (32.806671, -86.791130), "AK": (61.370716, -152.404419),
    "AZ": (33.729759, -111.431221), "AR": (34.969704, -92.373123),
    "CA": (36.116203, -119.681564), "CO": (39.059811, -105.311104),
    "CT": (41.597782, -72.755371), "DE": (39.318523, -75.507141),
    "FL": (27.766279, -81.686783), "GA": (33.040619, -83.643074),
    "HI": (21.094318, -157.498337), "ID": (44.240459, -114.478828),
    "IL": (40.349457, -88.986137), "IN": (39.849426, -86.258278),
    "IA": (42.011539, -93.210526), "KS": (38.526600, -96.726486),
    "KY": (37.668140, -84.670067), "LA": (31.169546, -91.867805),
    "ME": (44.693947, -69.381927), "MD": (39.063946, -76.802101),
    "MA": (42.230171, -71.530106), "MI": (43.326618, -84.536095),
    "MN": (45.694454, -93.900192), "MS": (32.741646, -89.678696),
    "MO": (38.456085, -92.288368), "MT": (46.921925, -110.454353),
    "NE": (41.125370, -98.268082), "NV": (38.313515, -117.055374),
    "NH": (43.452492, -71.563896), "NJ": (40.298904, -74.521011),
    "NM": (34.840515, -106.248482), "NY": (42.165726, -74.948051),
    "NC": (35.630066, -79.806419), "ND": (47.528912, -99.784012),
    "OH": (40.388783, -82.764915), "OK": (35.565342, -96.928917),
    "OR": (44.572021, -122.070938), "PA": (40.590752, -77.209755),
    "RI": (41.680893, -71.511780), "SC": (33.856892, -80.945007),
    "SD": (44.299782, -99.438828), "TN": (35.747845, -86.692345),
    "TX": (31.054487, -97.563461), "UT": (40.150032, -111.862434),
    "VT": (44.045876, -72.710686), "VA": (37.769337, -78.169968),
    "WA": (47.400902, -121.490494), "WV": (38.491226, -80.954453),
    "WI": (44.268543, -89.616508), "WY": (42.755966, -107.302490),
}


def _safe_slug(s: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in s).strip("_")


def _save_matplotlib_figure(fig, path: Path) -> None:
    if fig is None:
        return
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_plotly_figure(fig, path_base: Path) -> None:
    if fig is None:
        return
    try:
        fig.write_image(str(path_base.with_suffix(".png")))
    except Exception as err:
        print(f"[warn] Failed to save plotly PNG at {path_base.with_suffix('.png')}: {err}")


def _calc_subsidy_spend(pt: pd.DataFrame) -> float:
    if "subsidy_cf" not in pt.columns:
        return np.nan
    sub_cf = pd.to_numeric(pt["subsidy_cf"], errors="coerce").to_numpy(dtype=float)
    spend = float(TOTAL_MARKET_SIZE) * float(np.nansum(pt["s_cf"].to_numpy(dtype=float) * sub_cf))
    spend = spend * float(PRICE_SCALE_USD_PER_UNIT) / 1_000_000_000.0
    return spend


def _save_profit_change_vs_import_share(
    out: dict[str, pd.DataFrame],
    *,
    label_slug: str,
    fig_dir: Path,
    product_data: pd.DataFrame,
    costs_df2: pd.DataFrame,
    us_set: set[str],
) -> None:
    firm_tbl = out["firm_table"].copy()
    firm_tbl["pct_change_profit"] = 100.0 * firm_tbl["dpi_millions_usd"] / firm_tbl["pi0_millions_usd"]

    pt = out["product_table"].copy()
    if "plant_country" in pt.columns:
        pt = pt.drop(columns=["plant_country", "firm_ids"], errors="ignore")
    pd_map = product_data[["market_ids", "product_ids", "firm_ids", "plant_country"]].drop_duplicates()
    m = pt.merge(pd_map, on=["market_ids", "product_ids"], how="left")
    if "plant_country" not in m.columns:
        raise KeyError("plant_country not found after merge")
    m = m[m["plant_country"].astype(str).str.strip().eq("United States")].copy()

    pc_map = costs_df2[["product_ids", "pcUSCA_pct"]].drop_duplicates("product_ids")
    m = m.merge(pc_map, on="product_ids", how="left")
    m["import_share"] = 1.0 - pd.to_numeric(m["pcUSCA_pct"], errors="coerce")

    w = m["s0"].to_numpy(dtype=float)
    m["w"] = w
    firm_import = (
        m.groupby("firm_ids", dropna=False)
         .apply(lambda d: np.nan if d["w"].sum() == 0 else np.nansum(d["import_share"] * d["w"]) / np.nansum(d["w"]))
         .reset_index(name="import_share")
    )

    plot_df = firm_tbl.merge(firm_import, on="firm_ids", how="inner")
    plot_df = plot_df[np.isfinite(plot_df["import_share"])].copy()
    plot_df["import_share_pct"] = 100.0 * plot_df["import_share"]
    plot_df["is_us"] = plot_df["firm_ids"].astype(str).str.lower().isin(us_set)
    plot_df = plot_df[plot_df["is_us"]].copy()
    plot_df = plot_df[~plot_df["firm_ids"].astype(str).str.lower().eq("rivian")].copy()
    if plot_df.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(plot_df["import_share_pct"], plot_df["pct_change_profit"], color="0.2")
    for _, r in plot_df.iterrows():
        ax.annotate(
            str(r["firm_ids"]),
            (r["import_share_pct"], r["pct_change_profit"]),
            textcoords="offset points",
            xytext=(4, 2),
            ha="left",
            fontsize=8,
            color="black",
        )
    ax.set_xlabel("Parts import share for US-assembled vehicles (%)")
    ax.set_ylabel("Firm profit change (%)")
    plt.tight_layout()
    _save_matplotlib_figure(fig, fig_dir / f"profit_change_vs_import_share_{label_slug}.png")


def _find_existing_output_dir(out_root: Path, results_path: Path) -> Path | None:
    if not out_root.exists():
        return None
    matches: list[tuple[str, Path]] = []
    for p in out_root.iterdir():
        if not p.is_dir():
            continue
        meta_path = p / "metadata.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            continue
        meta_results = meta.get("results_file")
        if not meta_results:
            continue
        try:
            if Path(meta_results).resolve() == results_path.resolve():
                matches.append((meta.get("created_at", ""), p))
        except Exception:
            continue
    if not matches:
        return None
    matches.sort(key=lambda x: x[0])
    return matches[-1][1]


def _existing_outputs_ok(out_dir: Path) -> bool:
    if not (out_dir / "summary_tbl_all.csv.gz").exists():
        return False
    scenario_index_path = out_dir / "scenario_index.csv"
    if not scenario_index_path.exists():
        return False
    try:
        scenario_index = pd.read_csv(scenario_index_path)
    except Exception:
        return False
    for _, row in scenario_index.iterrows():
        slug = row.get("label_slug")
        if not slug:
            return False
        if not (out_dir / f"{slug}__product_table.csv.gz").exists():
            return False
        if not (out_dir / f"{slug}__firm_table.csv.gz").exists():
            return False
    return True


def main() -> None:
    # load results config
    cfg_path = Path("results_config.json")
    if not cfg_path.exists():
        cfg_path = Path("post_est") / "results_config.json"
    cfg = json.loads(cfg_path.read_text())
    results_path = Path(cfg["results_file"])
    if not results_path.is_absolute():
        results_path = (cfg_path.parent / results_path).resolve()

    ownership_mode = cfg.get("ownership_mode", "firm")
    owner_mapping_path = cfg.get("owner_mapping_path")
    allow_unmapped_brands = bool(cfg.get("allow_unmapped_brands", False))
    owner_map = None
    pricer_map = None
    owner_map_path = None
    if ownership_mode == "owner":
        if not owner_mapping_path:
            raise ValueError("ownership_mode='owner' requires owner_mapping_path in results_config.json")
        owner_map_path = Path(owner_mapping_path)
        if not owner_map_path.is_absolute():
            owner_map_path = (cfg_path.parent / owner_map_path).resolve()
        owner_map = load_owner_map(owner_map_path, owner_col="owner")
        pricer_map = load_pricer_map(owner_map_path)

    # load inputs
    results_name = results_path.name
    if "45W" in results_name:
        product_data_path = cfg.get("product_data_45W", "data/raw/product_data_45W.csv")
    else:
        product_data_path = cfg.get("product_data_subsidy", "data/raw/product_data_subsidy.csv")
    product_data_path = Path(product_data_path)
    if not product_data_path.is_absolute():
        product_data_path = (cfg_path.parent / product_data_path).resolve()
    if not product_data_path.exists():
        raise FileNotFoundError(f"product_data file not found: {product_data_path}")
    product_data = pd.read_csv(product_data_path)
    if "45W" in results_name:
        agent_data_path = cfg.get("agent_data_45W", "data/raw/agent_data_45W.csv")
    else:
        agent_data_path = cfg.get("agent_data_subsidy", "data/raw/agent_data.csv")
    agent_data_path = Path(agent_data_path)
    if not agent_data_path.is_absolute():
        agent_data_path = (cfg_path.parent / agent_data_path).resolve()
    if not agent_data_path.exists():
        raise FileNotFoundError(f"agent_data file not found: {agent_data_path}")
    agent_data = pd.read_csv(agent_data_path)
    agent_cf_path = Path("data/raw/agent_data_cf.csv")
    agent_data_cf = pd.read_csv(agent_cf_path) if agent_cf_path.exists() else None

    with open(results_path, "rb") as f:
        results = pickle.load(f)

    # determine output directory (reuse if possible)
    out_root = Path("saved_outputs")
    out_root.mkdir(exist_ok=True)
    existing_dir = _find_existing_output_dir(out_root, results_path)
    if existing_dir is not None:
        meta_path = existing_dir / "metadata.json"
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            meta = {}
        meta_owner_mode = meta.get("ownership_mode", "firm")
        meta_owner_path = meta.get("owner_mapping_path")
        if meta_owner_mode != ownership_mode:
            existing_dir = None
        elif ownership_mode == "owner" and owner_map_path is not None:
            if meta_owner_path != str(owner_map_path):
                existing_dir = None
    reuse_outputs = False
    if existing_dir is not None and _existing_outputs_ok(existing_dir):
        out_dir = existing_dir
        reuse_outputs = True
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_stem = _safe_slug(results_path.stem)
        out_dir = out_root / f"{results_stem}_{stamp}"
        out_dir.mkdir(parents=True, exist_ok=True)

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    if reuse_outputs:
        summary_tbl_all = pd.read_csv(out_dir / "summary_tbl_all.csv.gz", index_col=0)
        scenario_index = pd.read_csv(out_dir / "scenario_index.csv")
        scenario_rows = scenario_index.to_dict("records")
        outs = None
        product_data, _, costs_df2, _ = prepare_costs_df2_for_year(
            product_data=product_data,
            vehicle_costs_csv="data/derived/vehicle_costs_markups_chars.csv",
            pc_panel_csv="data/raw/pc_data_panel.csv",
            year=2024,
        )
        # regenerate figures only (no counterfactual solves)
        target_label = "parts and vehicles tariff (with subsidy)"
        target_out = None
        target_slug = None
        for row in scenario_rows:
            label = row["scenario_label"]
            label_slug = row["label_slug"]
            pt = pd.read_csv(out_dir / f"{label_slug}__product_table.csv.gz")
            ft = pd.read_csv(out_dir / f"{label_slug}__firm_table.csv.gz")
            out = {"product_table": pt, "firm_table": ft}
            if label == target_label:
                target_out = out
                target_slug = label_slug

            profit_art = build_profit_change_artifacts(
                out["firm_table"],
                scenario_label=label,
                us_firms=US_FIRMS,
            )
            _save_matplotlib_figure(profit_art.get("figure"), fig_dir / f"profit_changes_{label_slug}.png")

            pt_origin = out["product_table"].copy()
            if "plant_country" not in pt_origin.columns:
                pd_map = product_data[["market_ids", "product_ids", "plant_country"]].drop_duplicates(
                    ["market_ids", "product_ids"]
                )
                pt_origin = pt_origin.merge(pd_map, on=["market_ids", "product_ids"], how="left")
            if "plant_country" in pt_origin.columns:
                origin_tbl = origin_percent_metrics(pt_origin)
                plot_tbl = origin_tbl if "origin" not in origin_tbl.columns else origin_tbl.set_index("origin")
                fig = plot_origin_percent_metrics_bw(
                    plot_tbl,
                    title=None,
                    show=False,
                )
                _save_matplotlib_figure(fig, fig_dir / f"origin_metrics_{label_slug}.png")

            state_units = None
            try:
                state_units = build_state_units_table(
                    out["product_table"],
                    product_data,
                    total_market_size=TOTAL_MARKET_SIZE,
                )
            except Exception:
                state_units = None

            if state_units is not None and not state_units.empty:
                fig = build_state_map_figure(
                    state_units,
                    scenario_label=label,
                    state_abbr=STATE_ABBR,
                    state_centroids=STATE_CENTROIDS,
                )
                _save_plotly_figure(fig, fig_dir / f"assembly_map_{label_slug}")

            if agent_data_cf is not None:
                state_cs_tbl = None
                try:
                    state_cs_tbl = build_state_cs_table(
                        out,
                        agent_data_cf,
                        results=results,
                        market_id=CS_MARKET_ID,
                        price_x2_index=PRICE_X2_INDEX,
                        beta_price_index=PRICE_BETA_INDEX,
                        gamma=CS_GAMMA,
                    )
                except Exception:
                    state_cs_tbl = None
                if state_cs_tbl is not None and not state_cs_tbl.empty:
                    fig = build_state_cs_map_figure(
                        state_cs_tbl,
                        scenario_label=label,
                        state_abbr=STATE_ABBR,
                        state_centroids=STATE_CENTROIDS,
                    )
                    _save_plotly_figure(fig, fig_dir / f"cs_map_{label_slug}")

        if target_out is not None and target_slug is not None:
            us_set = {f.lower() for f in US_FIRMS}
            _save_profit_change_vs_import_share(
                target_out,
                label_slug=target_slug,
                fig_dir=fig_dir,
                product_data=product_data,
                costs_df2=costs_df2,
                us_set=us_set,
            )

        pd.set_option("display.max_columns", None)
        pd.set_option("display.max_rows", None)
        pd.set_option("display.width", 200)

        print("\nMAIN OUTPUT TABLE (summary_tbl_all)\n")
        print(summary_tbl_all.to_string())
        print("\nSaved outputs to:", out_dir)
        return
    else:
        # prepare costs
        product_data, rho_data, costs_df2, diag = prepare_costs_df2_for_year(
            product_data=product_data,
            vehicle_costs_csv="data/derived/vehicle_costs_markups_chars.csv",
            pc_panel_csv="data/raw/pc_data_panel.csv",
            year=2024,
        )
        vehicle_costs_df = pd.read_csv("data/derived/vehicle_costs_markups_chars.csv")
        if ownership_mode == "owner":
            missing_cols = [c for c in ["owner_ids", "pricer_ids"] if c not in vehicle_costs_df.columns]
            if missing_cols:
                raise ValueError(
                    "ownership_mode='owner' requires owner_ids and pricer_ids in vehicle_costs_markups_chars.csv. "
                    "Regenerate costs via get_elas_div.ipynb."
                )
        costs_full = build_costs_vector_from_vehicle_costs(results, vehicle_costs_df)

    # run counterfactuals (suppress verbose pyblp output)
    specs = default_scenario_specs(
        parts_tariff=PARTS_TARIFF,
        vehicle_tariff=VEHICLE_TARIFF,
        country_tariffs=COUNTRY_TARIFFS,
    )

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        outs = run_scenario_outputs(
            results,
            product_data,
            costs_df2,
            agent_data=agent_data,
            costs_full=costs_full,
            ownership_mode=ownership_mode,
            owner_map=owner_map,
            pricer_map=pricer_map,
            allow_unmapped_brands=allow_unmapped_brands,
            year=2024,
            price_x2_index=PRICE_X2_INDEX,
            beta_price_index=PRICE_BETA_INDEX,
            gamma=CS_GAMMA,
            specs=specs,
        )

    # EV share + tariff revenue table
    scenarios_for_ev = {}
    for key, meta in outs.items():
        spec = specs[key]
        scenarios_for_ev[meta["label"]] = {
            "out": meta["out"],
            "vehicle_tariff": spec.vehicle_tariff,
            "parts_tariff": spec.parts_tariff,
            "country_tariffs": spec.country_tariffs,
        }

    ev_tariff_tbl = build_ev_and_tariff_table(
        scenarios_for_ev,
        product_data=product_data,
        costs_df2=costs_df2,
        market_id=2024,
        total_market_size=TOTAL_MARKET_SIZE,
        price_scale_usd_per_unit=PRICE_SCALE_USD_PER_UNIT,
    )

    baseline_label = "no tariff (with subsidy)"
    baseline_meta = next((m for m in outs.values() if m["label"] == baseline_label), None)
    if baseline_meta is None:
        raise ValueError(f"Baseline scenario not found: {baseline_label}")
    baseline_subsidy_spend = _calc_subsidy_spend(baseline_meta["out"]["product_table"])
    if not np.isfinite(baseline_subsidy_spend):
        raise ValueError("Baseline subsidy spend could not be computed (missing subsidy_cf).")

    # main summary table
    summary_rows = []
    _ev_map = {r["Scenario"]: r for _, r in ev_tariff_tbl.iterrows()}

    us_set = {f.lower() for f in US_FIRMS}

    for key, meta in outs.items():
        label = meta["label"]
        out = meta["out"]
        pt = out["product_table"].copy()
        ft = out["firm_table"].copy()

        w0 = pt["s0"].to_numpy(dtype=float)
        w0_sum = float(np.nansum(w0)) if np.isfinite(w0).any() else np.nan
        dp_sw = np.nan if w0_sum == 0 else float(np.nansum(pt["dp_pct"].to_numpy(dtype=float) * w0) / w0_sum)

        wcf = pt["s_cf"].to_numpy(dtype=float)
        wcf_sum = float(np.nansum(wcf)) if np.isfinite(wcf).any() else np.nan
        markup_sw = np.nan if wcf_sum == 0 else float(np.nansum(pt["margin_cf_pct"].to_numpy(dtype=float) * wcf) / wcf_sum)

        df_f = ft.copy()
        df_f["firm_lower"] = df_f["firm_ids"].astype(str).str.lower()
        us_mask = df_f["firm_lower"].isin(us_set)
        us_ps = float(df_f.loc[us_mask, "dpi_millions_usd"].sum()) / 1_000.0

        cs_tbl = cs_change_by_income_bins(
            results,
            pt,
            market_id=CS_MARKET_ID,
            price_x2_index=PRICE_X2_INDEX,
            beta_price_index=PRICE_BETA_INDEX,
            income_demo_index=INCOME_DEMO_INDEX,
            income_transform=INCOME_TRANSFORM,
            n_bins=5,
            gamma=CS_GAMMA,
        )
        cs_tbl = cs_tbl.rename(columns={"CS0": "cs0", "CS_cf": "cs_cf"})

        cs_scale = float(TOTAL_MARKET_SIZE) * float(PRICE_SCALE_USD_PER_UNIT) / 1_000_000.0
        cs_tbl["cs0_total"] = cs_tbl["cs0"] * cs_tbl["weight_mass"] * cs_scale
        cs_tbl["cs_cf_total"] = cs_tbl["cs_cf"] * cs_tbl["weight_mass"] * cs_scale
        cs_tbl["dcs_total"] = cs_tbl["cs_cf_total"] - cs_tbl["cs0_total"]

        cs_total = float(cs_tbl["dcs_total"].sum()) / 1_000.0
        cs_base_total = float(cs_tbl["cs0_total"].sum()) / 1_000.0
        cs_total_pct = np.nan if cs_base_total == 0 else 100.0 * cs_total / cs_base_total

        cs_quintile = {}
        for _, r in cs_tbl.iterrows():
            q = f"Q{int(r['bin'])}"
            d_cs = float(r["dcs_total"]) / 1_000.0
            pct = np.nan if r["cs0_total"] == 0 else 100.0 * d_cs / (float(r["cs0_total"]) / 1_000.0)
            cs_quintile[q] = f"{d_cs:.3f} ({pct:.1f}%)"

        total_sold = float(TOTAL_MARKET_SIZE) * (
            float(np.nansum(pt["s_cf"].to_numpy(dtype=float))) - float(np.nansum(pt["s0"].to_numpy(dtype=float)))
        ) / 1_000_000.0

        # avoid merge suffixes if plant_country already present
        pt = pt.drop(columns=[c for c in ["plant_country", "ev"] if c in pt.columns])
        cols = ["market_ids", "product_ids", "ev", "plant_country"]
        pd_map = product_data[cols].drop_duplicates(["market_ids", "product_ids"]).copy()
        m = pt.merge(pd_map, on=["market_ids", "product_ids"], how="left")
        if "plant_country" not in m.columns:
            raise KeyError("plant_country not found after merge; available columns: " + ",".join(m.columns))
        us_mask_prod = m["plant_country"].astype(str).str.strip().eq("United States")

        ev_share = float(np.nansum(m["s_cf"].to_numpy(dtype=float) * pd.to_numeric(m["ev"], errors="coerce").fillna(0.0).to_numpy(dtype=float)))
        ev_share = np.nan if np.nansum(m["s_cf"].to_numpy(dtype=float)) == 0 else 100.0 * ev_share / float(np.nansum(m["s_cf"].to_numpy(dtype=float)))

        us_share = float(np.nansum(m.loc[us_mask_prod, "s_cf"].to_numpy(dtype=float)))
        us_share = np.nan if np.nansum(m["s_cf"].to_numpy(dtype=float)) == 0 else us_share / float(np.nansum(m["s_cf"].to_numpy(dtype=float)))

        us_assembled = float(TOTAL_MARKET_SIZE) * (
            float(np.nansum(m.loc[us_mask_prod, "s_cf"].to_numpy(dtype=float))) - float(np.nansum(m.loc[us_mask_prod, "s0"].to_numpy(dtype=float)))
        ) / 1_000_000.0

        rev_row = _ev_map.get(label, {})
        tariff_rev = float(rev_row.get("Tariff revenue – total (million USD)", np.nan)) / 1_000.0

        subsidy_spend = _calc_subsidy_spend(pt)

        net_us_impact = us_ps + cs_total + tariff_rev - subsidy_spend + baseline_subsidy_spend

        summary_rows.append({
            "Scenario": label,
            "Sales-weighted Δ Price (%)": dp_sw,
            "Sales-weighted Markup (CF, %)": markup_sw,
            "US Producer Surplus (Δ, billion USD)": us_ps,
            "CS Δ total (billion USD)": f"{cs_total:.3f} ({cs_total_pct:.1f}%)" if np.isfinite(cs_total_pct) else f"{cs_total:.3f} (nan)",
            "CS Δ Q1 (billion USD)": cs_quintile.get("Q1"),
            "CS Δ Q2 (billion USD)": cs_quintile.get("Q2"),
            "CS Δ Q3 (billion USD)": cs_quintile.get("Q3"),
            "CS Δ Q4 (billion USD)": cs_quintile.get("Q4"),
            "CS Δ Q5 (billion USD)": cs_quintile.get("Q5"),
            "Δ vehicles sold (millions)": total_sold,
            "EV share of vehicles sold (CF, %)": ev_share,
            "US share of vehicles sold (CF)": us_share,
            "Δ US assembled (millions)": us_assembled,
            "Tariff revenue (billion USD)": tariff_rev,
            "EV subsidy spending (billion USD)": subsidy_spend,
            "Net US impact (billion USD)": net_us_impact,
        })

    summary_tbl_all = pd.DataFrame(summary_rows).set_index("Scenario").T

    # scenario ordering (match notebook)
    label_by_key = {k: v["label"] for k, v in outs.items()}
    order_labels = []

    for subsidy_zero in (False, True):
        for k, spec in specs.items():
            if spec.parts_tariff > 0 and spec.vehicle_tariff > 0 and spec.subsidy_zero == subsidy_zero:
                order_labels.append(label_by_key[k])

    for subsidy_zero in (False, True):
        for k, spec in specs.items():
            if spec.parts_tariff == 0 and spec.vehicle_tariff > 0 and spec.subsidy_zero == subsidy_zero:
                order_labels.append(label_by_key[k])

    for subsidy_zero in (True, False):
        for k, spec in specs.items():
            if spec.parts_tariff == 0 and spec.vehicle_tariff == 0 and spec.subsidy_zero == subsidy_zero:
                order_labels.append(label_by_key[k])

    order_labels = [l for l in order_labels if l in summary_tbl_all.columns]
    if order_labels:
        summary_tbl_all = summary_tbl_all[order_labels]

    row_order = [
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
    summary_tbl_all = summary_tbl_all.reindex(row_order)

    # ---- save outputs ----
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    metadata = {
        "created_at": stamp,
        "results_file": str(results_path),
        "results_stem": results_stem,
        "price_x2_index": PRICE_X2_INDEX,
        "price_beta_index": PRICE_BETA_INDEX,
        "cs_gamma": CS_GAMMA,
        "income_demo_index": INCOME_DEMO_INDEX,
        "income_transform": INCOME_TRANSFORM,
        "cs_market_id": CS_MARKET_ID,
        "parts_tariff": PARTS_TARIFF,
        "vehicle_tariff": VEHICLE_TARIFF,
        "country_tariffs": COUNTRY_TARIFFS,
        "total_market_size": TOTAL_MARKET_SIZE,
        "price_scale_usd_per_unit": PRICE_SCALE_USD_PER_UNIT,
        "costs_prep_diag": diag,
        "baseline_subsidy_spend_billion_usd": baseline_subsidy_spend,
        "scenarios": {k: v["label"] for k, v in outs.items()},
        "ownership_mode": ownership_mode,
        "owner_mapping_path": str(owner_map_path) if owner_map_path is not None else None,
        "ownership_pricer_column": "pricer" if ownership_mode == "owner" else None,
        "allow_unmapped_brands": allow_unmapped_brands,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    # save summary + EV/tariff tables
    summary_tbl_all.to_csv(out_dir / "summary_tbl_all.csv.gz", compression="gzip")
    ev_tariff_tbl.to_csv(out_dir / "ev_tariff_tbl.csv.gz", index=False, compression="gzip")

    # save per-scenario outputs
    scenario_index_rows = []
    for key, meta in outs.items():
        label = meta["label"]
        out = meta["out"]
        label_slug = _safe_slug(label)

        scenario_index_rows.append({
            "scenario_key": key,
            "scenario_label": label,
            "label_slug": label_slug,
        })

        # add provenance columns
        df_names = ["product_table", "firm_table", "market_surplus_table", "overall_surplus"]
        if out.get("owner_table") is not None:
            df_names.append("owner_table")
        for df_name in df_names:
            df = out[df_name].copy()
            df["results_file"] = str(results_path)
            df["scenario_label"] = label
            df.to_csv(out_dir / f"{label_slug}__{df_name}.csv.gz", index=False, compression="gzip")

        if out.get("cf_costs_df") is not None:
            df = out["cf_costs_df"].copy()
            df["results_file"] = str(results_path)
            df["scenario_label"] = label
            df.to_csv(out_dir / f"{label_slug}__cf_costs_df.csv.gz", index=False, compression="gzip")

        profit_art = build_profit_change_artifacts(
            out["firm_table"],
            scenario_label=label,
            us_firms=US_FIRMS,
        )
        _save_matplotlib_figure(profit_art.get("figure"), fig_dir / f"profit_changes_{label_slug}.png")

        pt_origin = out["product_table"].copy()
        if "plant_country" not in pt_origin.columns:
            pd_map = product_data[["market_ids", "product_ids", "plant_country"]].drop_duplicates(
                ["market_ids", "product_ids"]
            )
            pt_origin = pt_origin.merge(pd_map, on=["market_ids", "product_ids"], how="left")
        if "plant_country" in pt_origin.columns:
            origin_tbl = origin_percent_metrics(pt_origin)
            origin_tbl = origin_tbl.reset_index()
            origin_tbl["results_file"] = str(results_path)
            origin_tbl["scenario_label"] = label
            origin_tbl.to_csv(out_dir / f"{label_slug}__origin_metrics.csv.gz", index=False, compression="gzip")
            fig = plot_origin_percent_metrics_bw(
                origin_tbl.set_index("origin"),
                title=None,
                show=False,
            )
            _save_matplotlib_figure(fig, fig_dir / f"origin_metrics_{label_slug}.png")

        state_units = None
        try:
            state_units = build_state_units_table(
                out["product_table"],
                product_data,
                total_market_size=TOTAL_MARKET_SIZE,
            )
        except Exception:
            state_units = None

        if state_units is not None and not state_units.empty:
            state_units = state_units.copy()
            state_units["results_file"] = str(results_path)
            state_units["scenario_label"] = label
            state_units.to_csv(out_dir / f"{label_slug}__state_units.csv.gz", index=False, compression="gzip")
            fig = build_state_map_figure(
                state_units,
                scenario_label=label,
                state_abbr=STATE_ABBR,
                state_centroids=STATE_CENTROIDS,
            )
            _save_plotly_figure(fig, fig_dir / f"assembly_map_{label_slug}")

        if agent_data_cf is not None:
            state_cs_tbl = None
            try:
                state_cs_tbl = build_state_cs_table(
                    out,
                    agent_data_cf,
                    results=results,
                    market_id=CS_MARKET_ID,
                    price_x2_index=PRICE_X2_INDEX,
                    beta_price_index=PRICE_BETA_INDEX,
                    gamma=CS_GAMMA,
                )
            except Exception:
                state_cs_tbl = None
            if state_cs_tbl is not None and not state_cs_tbl.empty:
                state_cs_tbl = state_cs_tbl.copy()
                state_cs_tbl["results_file"] = str(results_path)
                state_cs_tbl["scenario_label"] = label
                state_cs_tbl.to_csv(out_dir / f"{label_slug}__state_cs.csv.gz", index=False, compression="gzip")
                fig = build_state_cs_map_figure(
                    state_cs_tbl,
                    scenario_label=label,
                    state_abbr=STATE_ABBR,
                    state_centroids=STATE_CENTROIDS,
                )
                _save_plotly_figure(fig, fig_dir / f"cs_map_{label_slug}")

    target_label = "parts and vehicles tariff (with subsidy)"
    target_meta = next((m for m in outs.values() if m["label"] == target_label), None)
    if target_meta is not None:
        target_slug = _safe_slug(target_label)
        us_set = {f.lower() for f in US_FIRMS}
        _save_profit_change_vs_import_share(
            target_meta["out"],
            label_slug=target_slug,
            fig_dir=fig_dir,
            product_data=product_data,
            costs_df2=costs_df2,
            us_set=us_set,
        )

    pd.DataFrame(scenario_index_rows).to_csv(out_dir / "scenario_index.csv", index=False)

    # print summary table
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 200)

    print("\nMAIN OUTPUT TABLE (summary_tbl_all)\n")
    print(summary_tbl_all.to_string())
    print("\nSaved outputs to:", out_dir)


if __name__ == "__main__":
    # Local import to keep module load side effects low
    import pickle
    main()
