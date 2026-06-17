#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

export PYTHON="${PYTHON:-python}"

printf '\n[1/4] Cost-side outputs and robustness\n'
bash cost_side/make.sh

printf '\n[2/4] Post-estimation counterfactuals and rebased bundle\n'
bash post_est/make.sh

printf '\n[3/4] Paper assets and reproduced PDF\n'
PAPER_ARGS=(--skip-render)
if [[ "${STRICT_CANONICAL_PDF:-0}" == "1" ]]; then
  PAPER_ARGS+=(--strict-canonical-pdf)
fi
bash paper/make.sh "${PAPER_ARGS[@]}"

printf '\n[4/4] Downstream consistency checks\n'
"$PYTHON" post_est/check_downstream_consistency.py

printf '\nDownstream replication pipeline complete.\n'
