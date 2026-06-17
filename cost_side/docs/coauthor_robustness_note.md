# Note to Co-authors: Cost-Side Robustness Checks

This note summarizes the exchange-rate identification robustness checks for the cost-side regression.

The short version is:

- the imported-parts cost channel remains visible on the canonical domestic panel;
- the canonical panel now excludes product-years with conflicting raw primary source countries, while allowing duplicate-row share averaging when the raw primary country agrees;
- the foreign-assembled placebo remains reassuring;
- the future-exchange-rate placebo remains active, so it is not a clean pass in either levels or first differences;
- country parsing and exchange-rate alignment are now shared across cost-side and post-estimation code.

## Canonical Panel

The cost-side panel builder now produces one canonical domestic regression panel: U.S.-assembled vehicles with a stable primary source country over time and no within-product-year disagreement in directly observed raw primary source country.

The builder writes `cost_side/outputs/primary_source_country_conflicts.csv` for excluded product-years. In the current build, the expected conflicts are the Ford Focus rows where raw primary source countries disagree between Germany and Mexico.

## Main Checks

The robustness workflow:

- rebuilds the cost-side panel with documented sample flags;
- runs foreign-assembled placebo regressions;
- runs future-RER and current-plus-future timing diagnostics;
- runs the canonical pass-through and RER-main-effect diagnostics;
- runs leave-one-country-out checks;
- runs alternative exposure definitions;
- decomposes the result across recovered costs, observed prices, and recovered markups.

The execution sequence and sample definitions are documented in:

- `cost_side/docs/robustness_execution_sequence.md`
- `cost_side/docs/panel_build_decisions.md`

## Interpretation

The foreign placebo and price/markup decomposition support the imported-parts cost channel. The binding concern remains timing: future exchange rates still predict current recovered costs after the sample-construction and exchange-rate fixes.

The paper should therefore present the cost-side evidence as supportive of the imported-parts mechanism, with the strongest interpretation coming from first differences and the foreign placebo, while avoiding a claim that the timing placebo fully validates the design.

## Relevant Output Files

- `cost_side/outputs/cost_reg_placebo_coefficients.csv`
- `cost_side/outputs/cost_reg_placebo_levels_table.tex`
- `cost_side/outputs/cost_reg_placebo_fd_table.tex`
- `cost_side/outputs/cost_reg_robustness_coefficients.csv`
- `cost_side/outputs/cost_reg_robustness_canonical_table.tex`
- `cost_side/outputs/cost_reg_robustness_rer_main_table.tex`
- `cost_side/outputs/leave_one_country_out_coefficients.csv`
- `cost_side/outputs/cost_reduced_form_fx.png`
