# paper_writeup.tex value update log

This log records value-only edits made in `paper_writeup.tex` to align prose/table numbers with latest generated outputs.

## Sources used
- `post_est/outputs/cf_summary_table.tex`
- `post_est/outputs/plant_location_changes.tex`
- `post_est/outputs/cs_by_state.tex`
- `cost_side/outputs/cost_reg_table.tex`
- `post_est/get_elas_div.ipynb` (printed metrics in notebook JSON)
- Latest CF firm tables in `post_est/saved_outputs/blp_results_20260202_060248_0111110110000011010_45w_dbl_20260210_095244/*__firm_table.csv.gz`
- Latest CF product table in `post_est/saved_outputs/blp_results_20260202_060248_0111110110000011010_45w_dbl_20260210_095244/no_tariff__with_subsidy__product_table.csv.gz`

## Value changes
- `paper_writeup.tex:875`
  - old: own-price elasticity `-6.75`, market elasticity `-0.41`
  - new: own-price elasticity `-6.764`, market elasticity `-0.420`
  - source: `post_est/get_elas_div.ipynb` printed metrics (`-6.764015`, `-0.419681...`)

- `paper_writeup.tex:877`
  - old: average markup `17.9%`
  - new: average markup `17.2%`
  - source: `post_est/get_elas_div.ipynb` printed metric (`0.172033`)

- `paper_writeup.tex:907`
  - old: cost pass-through sentence used `0.705%`
  - new: `0.715%`
  - source: `cost_side/outputs/cost_reg_table.tex` col (1) coefficient `0.715`

- `paper_writeup.tex:1149`
  - old: `eta = 0.705`
  - new: `eta = 0.715`
  - source: `cost_side/outputs/cost_reg_table.tex`

- `paper_writeup.tex:1204`
  - old: CS losses `$33.8b` and `$15.1b`
  - new: `$33.6b` and `$15.0b`
  - source: `post_est/outputs/cf_summary_table.tex`

- `paper_writeup.tex:1206`
  - old: tariff revenue `$42.4b`; losses `$33.8b` and `$2.07b`
  - new: tariff revenue `$41.9b`; losses `$33.6b` and `$2.42b`
  - source: `post_est/outputs/cf_summary_table.tex`

- `paper_writeup.tex:1210`
  - old: threshold phrase `more than $300m`; Honda value `$450m`
  - new: threshold phrase `more than $270m`; Honda value `$427m`
  - source: `parts_and_vehicles_tariff__with_subsidy__firm_table.csv.gz`

- `paper_writeup.tex:1222`
  - old: Ford `$1.3b`; Mercedes `-$519m`; BMW `-$393m`
  - new: Ford `$1.4b`; Mercedes `-$516m`; BMW `-$392m`
  - source: `parts_and_vehicles_tariff__with_subsidy__firm_table.csv.gz`

- `paper_writeup.tex:1233`
  - old: aggregate producer gain `$1.21b`; share rise `11.8 pp`; Ford `3%` vs `19%`; Jeep/Cadillac `12%` and `40%`; Tesla `19%` vs `12%`
  - new: aggregate producer gain `$1.31b`; share rise `13.7 pp`; Ford `2.5%` vs `20.8%`; Jeep/Cadillac `11.5%` and `38%`; Tesla `19.2%` vs `13.7%`
  - source: `post_est/outputs/cf_summary_table.tex` and `vehicles_only_tariff__with_subsidy__firm_table.csv.gz`

- `paper_writeup.tex:1244`
  - old: outside diversion `13.9%`; US-assembled inside share `51%`
  - new: outside diversion `19.3%`; US-assembled inside share `53.5%`
  - source: `post_est/outputs/diversions_top5.tex` (mean of six outside-good diversion entries) and `no_tariff__with_subsidy__product_table.csv.gz`

- `paper_writeup.tex:1283`
  - old: Michigan `7.1%`; Texas `4.0%`; California `13.1%`
  - new: Michigan `7.5%`; Texas `4.8%`; California `13.9%`
  - source: `post_est/outputs/plant_location_changes.tex`

- `paper_writeup.tex:1319`
  - old: price `10.14% -> 10.09%`; EV share `6.61% -> 4.48%`; decline `2.1 pp (~32%)`; extra sales drop `130,000` to `-1.02m`; no-tariff EV drop `25.9%`
  - new: price `10.06% -> 10.12%`; EV share `6.65% -> 3.15%`; decline `3.5 pp (~53%)`; extra sales drop `174,000` to `-1.10m`; no-tariff EV drop `54.5%`
  - source: `post_est/outputs/cf_summary_table.tex`

- `paper_writeup.tex:1321`
  - old: CS `-$36.4b (-10.5%)` vs `-$33.8b (-9.70%)`; incremental `$2.6b`
  - new: CS `-$37.6b (-11.5%)` vs `-$33.6b (-10.2%)`; incremental `$4.1b`
  - source: `post_est/outputs/cf_summary_table.tex`

- `paper_writeup.tex:1323`
  - old: Tesla `57%`; US producer surplus `-$2.07b -> -$4.14b`
  - new: Tesla `53%`; US producer surplus `-$2.42b -> -$4.38b`
  - source: `no_tariff__no_subsidy__firm_table.csv.gz` and `post_est/outputs/cf_summary_table.tex`

- `paper_writeup.tex:1348`
  - old: typical state CS range `0.3%---1.0%`; OR `-2.4%`; WA `-1.4%`; CA `-1.3%`; AK `-2.0%`
  - new: typical state CS range `0.6%---1.8%`; OR `-4.6%`; WA `-3.0%`; CA `-3.1%`; AK `-3.9%`
  - source: `post_est/outputs/cs_by_state.tex` (no tariff, no subsidy column)

- `paper_writeup.tex:1544`
  - old: vehicle-only summary prose `5.76%`, `$15.1b (4.4%)`, `$1.21b`, `53.5% -> 67.3%`
  - new: `5.70%`, `$15.0b (4.6%)`, `$1.31b`, `53.5% -> 67.2%`
  - source: `post_est/outputs/cf_summary_table.tex`

- `paper_writeup.tex:1555`
  - old table values: `5.76`, `-15.1 (-4.40%)`, `1.21`, `6.78`, `67.3`
  - new table values: `5.70`, `-15.0 (-4.60%)`, `1.31`, `6.75`, `67.2`
  - source: `post_est/outputs/cf_summary_table.tex`

- `paper_writeup.tex:1590`
  - old: parts+vehicles summary prose `10.14%`, `$33.8b (9.7%)`, `$2.07b`, `$42.4b`
  - new: `10.06%`, `$33.6b (10.2%)`, `$2.42b`, `$41.9b`
  - source: `post_est/outputs/cf_summary_table.tex`

- `paper_writeup.tex:1601`
  - old table values: `10.14`, `-33.8 (-9.70%)`, `-2.07`, `6.61`, `57.8`, `42.4`
  - new table values: `10.06`, `-33.6 (-10.2%)`, `-2.42`, `6.65`, `57.6`, `41.9`
  - source: `post_est/outputs/cf_summary_table.tex`

- `paper_writeup.tex:1651`
  - old: EV share `6.61 -> 4.48` (`~32%`), incremental CS loss `$2.6b`
  - new: EV share `6.65 -> 3.15` (`~53%`), incremental CS loss `$4.1b`
  - source: `post_est/outputs/cf_summary_table.tex`

## Suggestions (claims that may no longer hold)
- `paper_writeup.tex:1210`
  - Current text still says Honda is the largest beneficiary in the full-tariff-with-subsidy scenario.
  - Updated firm table values indicate Tesla (`+$457.9m`) exceeds Honda (`+$427.0m`).

- `paper_writeup.tex:1233`
  - Phrase says “US firms' share of sales rises by 13.7 pp.”
  - The 13.7 pp value is from US-assembled share in `cf_summary_table`; if this sentence intends US-headquartered firm share, this should be checked and possibly relabeled.

- `paper_writeup.tex:1319`
  - Sentence still says no-tariff EV-share fall of `54.5%` is “close to” Allcott et al. `26.7%`.
  - This closeness claim likely no longer holds.

