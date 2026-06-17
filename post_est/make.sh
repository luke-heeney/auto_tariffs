#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-python}"
export RESULTS_CONFIG_PATH="${RESULTS_CONFIG_PATH:-post_est/results_config.json}"

printf 'Building post-estimation tables from fixed BLP results...\n'
"$PYTHON" post_est/build_blp_est_table.py
"$PYTHON" post_est/build_micro_moments_table.py
"$PYTHON" post_est/build_ev_subsidy_tables.py

printf 'Running canonical counterfactual batch with %s...\n' "$RESULTS_CONFIG_PATH"
"$PYTHON" post_est/run_cf_batch.py

printf 'Rebasing counterfactual outputs to B0: no tariff, no subsidy...\n'
"$PYTHON" post_est/rebase_saved_outputs_b0.py

CANONICAL_BUNDLE="$("$PYTHON" post_est/check_downstream_consistency.py --print-canonical-bundle)"
printf 'Exporting graph values from %s...\n' "$CANONICAL_BUNDLE"
env CF_SOURCE_BUNDLE="$CANONICAL_BUNDLE" "$PYTHON" post_est/export_profit_change_graph_values.py

printf 'Checking post-estimation outputs...\n'
"$PYTHON" post_est/check_downstream_consistency.py --skip-cost-side --skip-paper
