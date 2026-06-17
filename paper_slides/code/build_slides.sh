#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLIDES_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SLIDES_DIR}/.." && pwd)"
OUT_DIR="${SLIDES_DIR}/output"
ASSET_DIR="${OUT_DIR}/assets"

mkdir -p "${ASSET_DIR}"

copy_asset() {
  local name="$1"
  cp "${REPO_ROOT}/paper/generated/graphs/${name}" "${ASSET_DIR}/${name}"
}

copy_asset "domestic_share.png"
copy_asset "origin_metrics_vehicles_only_tariff__no_subsidy.png"
copy_asset "origin_metrics_parts_and_vehicles_tariff__no_subsidy.png"
copy_asset "z_scatter_pct_more.png"
copy_asset "profit_changes_vehicles_only_tariff__no_subsidy.png"
copy_asset "profit_changes_parts_and_vehicles_tariff__no_subsidy.png"
copy_asset "profit_change_vs_import_share_parts_and_vehicles_tariff__with_subsidy.png"
copy_asset "profit_changes_no_tariff__with_subsidy.png"
copy_asset "cs_map_no_tariff__with_subsidy.png"
copy_asset "assembly_map_vehicles_only_tariff__no_subsidy.png"
copy_asset "assembly_map_parts_and_vehicles_tariff__no_subsidy.png"
copy_asset "price_coef.png"
copy_asset "markups_dist.png"
copy_asset "ev_share_sales_by_scenario.png"
copy_asset "ev_share_sales_by_scenario_compact.png"
copy_asset "cs_quintile_grouped_bars.png"
copy_asset "ev_firm_sales_change_by_scenario.png"
copy_asset "ev_firm_sales_change_c1_c3.png"

for name in \
  "news_cnbc_gm.png" \
  "news_reuters_prices.png" \
  "news_wsj_ford.png" \
  "news_ap_tariffs.png"; do
  cp "${SCRIPT_DIR}/assets/${name}" "${ASSET_DIR}/${name}"
done

latexmk \
  -pdf \
  -interaction=nonstopmode \
  -halt-on-error \
  -outdir="${OUT_DIR}" \
  "${SCRIPT_DIR}/auto_tariffs_seminar.tex"

echo "Built ${OUT_DIR}/auto_tariffs_seminar.pdf"
