# Downstream Audit Findings

## Findings addressed

- Added a root `make.sh` so the downstream replication path has one primary
  command.
- Added `cost_side/make.sh`, `post_est/make.sh`, and `paper/make.sh` so each
  module has a consistent entry point.
- Added `post_est/check_downstream_consistency.py` to validate canonical config
  paths, bundle metadata, scenario labels, generated files, B0 rebasing
  invariants, cost-side outputs, and paper manifest consistency.
- Updated the Makefile to delegate to the new shell wrappers instead of
  maintaining a second independent orchestration path.
- Documented the canonical assumptions and separated the default paper path from
  alternate counterfactual configs.
- Made the 2025 IMF exchange-rate backfill reproducible offline by using the
  cached values in `cost_side/outputs/exchange_rate_2025_backfill_values.csv`
  by default; live IMF refresh now requires `--refresh`.
- Added a Matplotlib static-map fallback for state-level Plotly maps so map PNGs
  can still be generated when Kaleido or headless Chrome fails locally.

## Audit issues to watch

- The repository contains many historical saved-output directories. The new
  checker selects the canonical bundle by metadata instead of a hardcoded
  timestamp, but historical bundles are intentionally preserved.
- `post_est/results_config_elasticity_interaction.json` is a useful robustness
  path, but it is not the canonical paper path unless explicitly promoted.
- `paper/build_paper.py` can still run upstream rendering by itself. In the root
  wrapper, paper build is called with `--skip-render` to avoid re-running
  counterfactuals after `post_est/make.sh`.
- The canonical EV share uses `engine_type == "Electric"`. PHEVs are folded into
  `Hybrid` in the underlying product data and need an explicit proxy if a
  plug-in vehicle share is reported.
- Some generated outputs are tracked or already dirty in the current worktree.
  The new pipeline does not delete, reset, or normalize those files.
- Plotly/Kaleido map PNG export is unstable in the current macOS environment.
  The scripts now fall back to a deterministic Matplotlib state-label map and
  still emit HTML fallbacks when static export is unavailable.

## Residual risk

- Full end-to-end runtime may be long because the cost-side robustness report
  compiles multiple R scripts and LaTeX reports.
- Exact canonical PDF text matching is still controlled by
  `--strict-canonical-pdf`; the standard pipeline requires a successful build
  and asset validation, not exact text identity.
- Historical saved-output bundles are preserved by design, so cleanup remains an
  explicit maintenance task rather than part of replication.
