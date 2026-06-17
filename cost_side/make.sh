#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-python}"
RSCRIPT="${RSCRIPT:-Rscript}"

printf 'Building cost-side panel and primary cost-side outputs...\n'
"$PYTHON" cost_side/build_cost_side_panel.py
"$RSCRIPT" cost_side/cost_reg.R
"$RSCRIPT" cost_side/cost_reg_elas.R

printf 'Building cost-side robustness reports...\n'
bash cost_side/build_robustness_report.sh
