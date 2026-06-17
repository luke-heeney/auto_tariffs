source("cost_side/robustness_helpers.R")

out_dir <- ensure_outputs_dir()

canonical_panel <- load_cost_panel("cost_side_panel.csv")

sample_counts <- sample_summary(canonical_panel, "canonical_domestic_panel")
write.csv(
  sample_counts,
  file.path(out_dir, "cost_reg_robustness_sample_counts.csv"),
  row.names = FALSE
)

canonical_df <- build_regression_frame(canonical_panel, include_elasticities = FALSE) %>%
  filter(!is.na(pcOth1_pct1_lag1))

canonical_elas_df <- build_regression_frame(canonical_panel, include_elasticities = TRUE) %>%
  filter(!is.na(pcOth1_pct1_lag1))

mL_base <- feols(
  ln_costs ~ ln_inv_rer_code1:pcOth1_pct1_lag1 +
    ln_size + ln_weight + ln_hp + ln_mpg | make_model + year,
  data = canonical_df,
  cluster = ~ make_model
)

canonical_fd <- build_fd_frame(canonical_df, keep_cols = c("pcOth1_pct1_lag1"))

mFD_base <- feols(
  ln_costs ~ ln_inv_rer_code1:pcOth1_pct1_lag1 +
    ln_size + ln_weight + ln_hp + ln_mpg | year,
  data = canonical_fd,
  cluster = ~ make_model
)

mL_rer_main <- feols(
  ln_costs ~ ln_inv_rer_code1 +
    ln_inv_rer_code1:pcOth1_pct1_lag1 +
    ln_size + ln_weight + ln_hp + ln_mpg | make_model + year,
  data = canonical_df,
  cluster = ~ make_model
)

mFD_rer_main <- feols(
  ln_costs ~ ln_inv_rer_code1 +
    ln_inv_rer_code1:pcOth1_pct1_lag1 +
    ln_size + ln_weight + ln_hp + ln_mpg | year,
  data = canonical_fd,
  cluster = ~ make_model
)

mL_elas <- feols(
  ln_costs ~ ln_inv_rer_code1:pcOth1_pct1_lag1 +
    ln_inv_rer_code1:pcOth1_pct1_lag1:log_abs_own_elas_lag1 +
    ln_size + ln_weight + ln_hp + ln_mpg | make_model + year,
  data = canonical_elas_df,
  cluster = ~ make_model
)

canonical_elas_fd <- build_fd_frame(
  canonical_elas_df,
  keep_cols = c("pcOth1_pct1_lag1", "log_abs_own_elas_lag1")
)

mFD_elas <- feols(
  ln_costs ~ ln_inv_rer_code1:pcOth1_pct1_lag1 +
    ln_inv_rer_code1:pcOth1_pct1_lag1:log_abs_own_elas_lag1 +
    ln_size + ln_weight + ln_hp + ln_mpg | year,
  data = canonical_elas_fd,
  cluster = ~ make_model
)

dict <- c(
  "ln_inv_rer_code1" = "$\\log(RER_{jt})$",
  "ln_inv_rer_code1:pcOth1_pct1_lag1" = "$\\rho_{j,t-1}\\times\\log(RER_{jt})$",
  "pcOth1_pct1_lag1:ln_inv_rer_code1" = "$\\rho_{j,t-1}\\times\\log(RER_{jt})$",
  "ln_inv_rer_code1:pcOth1_pct1_lag1:log_abs_own_elas_lag1" = "$\\rho_{j,t-1}\\times\\log|\\varepsilon_{j,t-1}|\\times\\log(RER_{jt})$",
  "ln_size" = "$\\ln(\\text{size})$",
  "ln_weight" = "$\\ln(\\text{weight})$",
  "ln_hp" = "$\\ln(\\text{hp})$",
  "ln_mpg" = "$\\ln(\\text{mpg})$"
)

canonical_table_path <- file.path(out_dir, "cost_reg_robustness_canonical_table.tex")
rer_main_table_path <- file.path(out_dir, "cost_reg_robustness_rer_main_table.tex")
elas_table_path <- file.path(out_dir, "cost_reg_robustness_elas_table.tex")

etable(
  mL_base, mFD_base,
  tex = TRUE,
  file = canonical_table_path,
  replace = TRUE,
  title = "Baseline pass-through estimates",
  label = "tab:cost_reg_robustness_canonical",
  dict = dict,
  fitstat = ~ n + r2 + wr2,
  signif.code = c("***" = 0.01, "**" = 0.05, "*" = 0.10)
)

etable(
  mL_rer_main, mFD_rer_main,
  tex = TRUE,
  file = rer_main_table_path,
  replace = TRUE,
  title = "Pass-through robustness with an exchange-rate main effect",
  label = "tab:cost_reg_robustness_rer_main",
  dict = dict,
  fitstat = ~ n + r2 + wr2,
  signif.code = c("***" = 0.01, "**" = 0.05, "*" = 0.10)
)

etable(
  mL_elas, mFD_elas,
  tex = TRUE,
  file = elas_table_path,
  replace = TRUE,
  title = "Elasticity-interaction diagnostics",
  label = "tab:cost_reg_robustness_elas",
  dict = dict,
  fitstat = ~ n + r2 + wr2,
  signif.code = c("***" = 0.01, "**" = 0.05, "*" = 0.10)
)

relabel_model_numbers(canonical_table_path, c("(1)", "(2)"))
relabel_model_numbers(rer_main_table_path, c("(1)", "(2)"))
relabel_model_numbers(elas_table_path, c("(1)", "(2)"))

append_table_note(
  canonical_table_path,
  "The dependent variable is log recovered marginal cost. Column (1) is the levels specification with make-model and year fixed effects. Column (2) is the first-difference specification with year fixed effects; the smaller sample reflects the requirement that a make-model be observed in consecutive years with non-missing differenced costs, exchange rates, exposure, and controls. The exchange-rate term is interacted with lagged imported-parts exposure. Vehicle controls are log size, log weight, log horsepower, and log miles per gallon. Standard errors are clustered by make-model."
)
append_table_note(
  rer_main_table_path,
  "The dependent variable is log recovered marginal cost. Column (1) is the levels specification with make-model and year fixed effects. Column (2) is the first-difference specification with year fixed effects. Both columns add an unrestricted source-country exchange-rate term to the exposure interaction. Vehicle controls are log size, log weight, log horsepower, and log miles per gallon. Standard errors are clustered by make-model."
)
append_table_note(
  elas_table_path,
  "The dependent variable is log recovered marginal cost. Column (1) is the levels specification with make-model and year fixed effects. Column (2) is the first-difference specification with year fixed effects. The table augments the baseline exposure interaction with an interaction between lagged imported-parts exposure, the exchange-rate term, and lagged log absolute own-price elasticity. Standard errors are clustered by make-model."
)

coef_rows <- bind_rows(
  extract_term_row(mL_base, "ln_inv_rer_code1:pcOth1_pct1_lag1", "canonical_domestic", "levels_baseline"),
  extract_term_row(mFD_base, "ln_inv_rer_code1:pcOth1_pct1_lag1", "canonical_domestic", "fd_baseline"),
  extract_term_row(mL_rer_main, "ln_inv_rer_code1", "canonical_domestic", "levels_rer_main"),
  extract_term_row(mL_rer_main, "ln_inv_rer_code1:pcOth1_pct1_lag1", "canonical_domestic", "levels_rer_main"),
  extract_term_row(mFD_rer_main, "ln_inv_rer_code1", "canonical_domestic", "fd_rer_main"),
  extract_term_row(mFD_rer_main, "ln_inv_rer_code1:pcOth1_pct1_lag1", "canonical_domestic", "fd_rer_main"),
  extract_term_row(mL_elas, "ln_inv_rer_code1:pcOth1_pct1_lag1", "canonical_domestic", "levels_elas"),
  extract_term_row(mL_elas, "ln_inv_rer_code1:pcOth1_pct1_lag1:log_abs_own_elas_lag1", "canonical_domestic", "levels_elas"),
  extract_term_row(mFD_elas, "ln_inv_rer_code1:pcOth1_pct1_lag1", "canonical_domestic", "fd_elas"),
  extract_term_row(mFD_elas, "ln_inv_rer_code1:pcOth1_pct1_lag1:log_abs_own_elas_lag1", "canonical_domestic", "fd_elas")
)

write.csv(
  coef_rows,
  file.path(out_dir, "cost_reg_robustness_coefficients.csv"),
  row.names = FALSE
)

rer_main_coef_rows <- bind_rows(
  extract_term_row(mL_rer_main, "ln_inv_rer_code1", "canonical_domestic", "levels_rer_main"),
  extract_term_row(mL_rer_main, "ln_inv_rer_code1:pcOth1_pct1_lag1", "canonical_domestic", "levels_rer_main"),
  extract_term_row(mFD_rer_main, "ln_inv_rer_code1", "canonical_domestic", "fd_rer_main"),
  extract_term_row(mFD_rer_main, "ln_inv_rer_code1:pcOth1_pct1_lag1", "canonical_domestic", "fd_rer_main")
)

write.csv(
  rer_main_coef_rows,
  file.path(out_dir, "cost_reg_robustness_rer_main_coefficients.csv"),
  row.names = FALSE
)

notes_lines <- c(
  "# Cost-Reg Robustness Notes",
  "",
  "This file is generated by `cost_side/cost_reg_robustness.R`.",
  "",
  "## Sample Definition",
  "",
  "- `canonical_domestic`: U.S.-assembled vehicles with a stable primary source country over time and no within-product-year disagreement in raw primary source country.",
  "",
  "## Specifications",
  "",
  "- The canonical table reports the standard pass-through regression in levels and first differences.",
  "- The main-RER table adds an unrestricted exchange-rate main effect alongside the exposure interaction on the canonical domestic sample.",
  "- The elasticity table re-estimates the lagged log-absolute-elasticity interaction used by the optional counterfactual path.",
  "",
  "## Output Files",
  "",
  "- `cost_reg_robustness_canonical_table.tex`",
  "- `cost_reg_robustness_rer_main_table.tex`",
  "- `cost_reg_robustness_elas_table.tex`",
  "- `cost_reg_robustness_coefficients.csv`",
  "- `cost_reg_robustness_rer_main_coefficients.csv`",
  "- `cost_reg_robustness_sample_counts.csv`"
)
writeLines(notes_lines, file.path(out_dir, "cost_reg_robustness_notes.md"))

cat("Saved:\n")
cat(" -", canonical_table_path, "\n")
cat(" -", rer_main_table_path, "\n")
cat(" -", elas_table_path, "\n")
cat(" -", file.path(out_dir, "cost_reg_robustness_coefficients.csv"), "\n")
cat(" -", file.path(out_dir, "cost_reg_robustness_rer_main_coefficients.csv"), "\n")
