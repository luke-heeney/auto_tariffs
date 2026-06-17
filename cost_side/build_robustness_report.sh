#!/usr/bin/env bash
set -euo pipefail

python cost_side/build_cost_side_panel.py
python cost_side/backfill_exchange_rates_2025.py
Rscript cost_side/cost_reg_placebo.R
Rscript cost_side/cost_reg_timing_triplet.R
Rscript cost_side/cost_reg_robustness.R
Rscript cost_side/cost_reg_leave_one_country_out.R
Rscript cost_side/plot_cost_reduced_form.R
Rscript cost_side/cost_reg_price_markup_decomp.R
Rscript cost_side/cost_reg_alt_exposure.R
Rscript cost_side/cost_reg_vehicle_type.R
Rscript cost_side/plot_high_exposure_series.R
python cost_side/render_full_robustness_report_assets.py
python cost_side/render_cost_side_robustness_note_assets.py
xelatex -interaction=nonstopmode -output-directory=cost_side/docs cost_side/docs/full_robustness_report.tex
xelatex -interaction=nonstopmode -output-directory=cost_side/docs cost_side/docs/full_robustness_report.tex
xelatex -interaction=nonstopmode -output-directory=cost_side/docs cost_side/docs/cost_side_robustness_note.tex
xelatex -interaction=nonstopmode -output-directory=cost_side/docs cost_side/docs/cost_side_robustness_note.tex

printf 'Built %s\n' "cost_side/docs/full_robustness_report.pdf"
printf 'Built %s\n' "cost_side/docs/cost_side_robustness_note.pdf"
