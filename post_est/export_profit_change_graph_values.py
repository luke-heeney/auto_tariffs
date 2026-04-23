from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from run_cf_batch import US_FIRMS


def _latest_complete_saved_output_dir() -> Path:
    base = Path("saved_outputs")
    candidates: list[tuple[str, float, Path]] = []
    for d in base.iterdir():
        if not d.is_dir():
            continue
        meta_path = d / "metadata.json"
        scenario_index_path = d / "scenario_index.csv"
        summary_path = d / "summary_tbl_all.csv.gz"
        if not (meta_path.exists() and scenario_index_path.exists() and summary_path.exists()):
            continue
        try:
            meta = json.loads(meta_path.read_text())
            created_at = str(meta.get("created_at", ""))
        except Exception:
            created_at = ""
        candidates.append((created_at, d.stat().st_mtime, d))
    if not candidates:
        raise FileNotFoundError("No complete saved output directories found under post_est/saved_outputs.")
    candidates.sort(key=lambda x: (x[0], x[1], x[2].name))
    return candidates[-1][2]


def main() -> None:
    src_override = os.environ.get("CF_SOURCE_BUNDLE")
    if src_override:
        bundle_dir = Path(src_override)
    else:
        bundle_dir = _latest_complete_saved_output_dir()

    scenario_index = pd.read_csv(bundle_dir / "scenario_index.csv")
    us_set = {firm.lower() for firm in US_FIRMS}
    rows: list[pd.DataFrame] = []

    for _, scenario in scenario_index.iterrows():
        slug = str(scenario["label_slug"])
        scenario_key = str(scenario["scenario_key"])
        scenario_label = str(scenario["scenario_label"])

        firm_table = pd.read_csv(bundle_dir / f"{slug}__firm_table.csv.gz").copy()
        firm_table["scenario_key"] = scenario_key
        firm_table["scenario_label"] = scenario_label
        firm_table["label_slug"] = slug
        firm_table["firm_ids"] = firm_table["firm_ids"].astype(str)
        firm_table["plotted_firm_label"] = firm_table["firm_ids"].replace({"mercedesbenz": "mercedes"})
        firm_table["firm_lower"] = firm_table["firm_ids"].str.lower()
        firm_table["is_us"] = firm_table["firm_lower"].isin(us_set)
        base = pd.to_numeric(firm_table["pi0_millions_usd"], errors="coerce").to_numpy(dtype=float)
        dlt = pd.to_numeric(firm_table["dpi_millions_usd"], errors="coerce").to_numpy(dtype=float)
        pct = np.full(len(firm_table), np.nan, dtype=float)
        ok = np.isfinite(base) & (base != 0)
        pct[ok] = 100.0 * dlt[ok] / base[ok]
        firm_table["pct_change_profit"] = pct
        firm_table = firm_table.sort_values("dpi_millions_usd", ascending=False, kind="mergesort").reset_index(drop=True)
        firm_table["plot_rank"] = np.arange(1, len(firm_table) + 1)

        n_label = 6
        if len(firm_table) <= 2 * n_label:
            annotate_idx = set(range(len(firm_table)))
        else:
            annotate_idx = set(range(n_label)) | set(range(len(firm_table) - n_label, len(firm_table)))
        firm_table["annotated_in_graph"] = [i in annotate_idx for i in range(len(firm_table))]

        rows.append(
            firm_table[
                [
                    "scenario_key",
                    "scenario_label",
                    "label_slug",
                    "plot_rank",
                    "firm_ids",
                    "plotted_firm_label",
                    "is_us",
                    "annotated_in_graph",
                    "share0_total",
                    "pi0_millions_usd",
                    "pi_cf_millions_usd",
                    "dpi_millions_usd",
                    "pct_change_profit",
                ]
            ]
            .rename(columns={"share0_total": "market_share"})
        )

    out = pd.concat(rows, ignore_index=True)
    out_path = bundle_dir / "profit_changes_graph_values.csv"
    out.to_csv(out_path, index=False)
    print(f"Source bundle: {bundle_dir}")
    print(f"Saved graph values CSV: {out_path}")


if __name__ == "__main__":
    main()
