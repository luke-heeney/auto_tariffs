source("cost_side/robustness_helpers.R")

build_fd_future_rer_frame <- function(df) {
  df %>%
    arrange(make_model, year) %>%
    group_by(make_model) %>%
    mutate(
      year_gap = year - lag(year),
      d_ln_costs = ln_costs - lag(ln_costs),
      d_ln_inv_rer_code1 = ln_inv_rer_code1 - lag(ln_inv_rer_code1),
      d_ln_size = ln_size - lag(ln_size),
      d_ln_weight = ln_weight - lag(ln_weight),
      d_ln_hp = ln_hp - lag(ln_hp),
      d_ln_mpg = ln_mpg - lag(ln_mpg),
      d_ln_inv_rer_code1_tplus1 = ln_inv_rer_code1_tplus1 - ln_inv_rer_code1
    ) %>%
    ungroup() %>%
    filter(year_gap == 1) %>%
    filter(
      !is.na(d_ln_costs),
      !is.na(d_ln_inv_rer_code1),
      !is.na(d_ln_inv_rer_code1_tplus1)
    ) %>%
    transmute(
      make_model = make_model,
      year = year,
      pcOth1_code1 = pcOth1_code1,
      ln_costs = d_ln_costs,
      ln_inv_rer_code1 = d_ln_inv_rer_code1,
      d_ln_inv_rer_code1_tplus1 = d_ln_inv_rer_code1_tplus1,
      pcOth1_pct1_lag1 = pcOth1_pct1_lag1,
      ln_size = d_ln_size,
      ln_weight = d_ln_weight,
      ln_hp = d_ln_hp,
      ln_mpg = d_ln_mpg
    )
}

out_dir <- ensure_outputs_dir()
all_panel <- load_cost_panel("cost_side_panel_all.csv")
exchange_lookup <- load_normalized_exchange_lookup()

domestic_panel <- all_panel %>%
  filter(is_us_assembled, canonical_exposure_ok)

foreign_panel <- all_panel %>%
  filter(is_foreign_assembled, source_country_stable)

domestic_df <- build_regression_frame(domestic_panel) %>%
  attach_direct_future_rer(
    exchange_lookup,
    code_col = "pcOth1_code1",
    year_col = "year",
    out_col = "ln_inv_rer_code1_tplus1"
  )
foreign_df <- build_regression_frame(foreign_panel)

domestic_current_rer_levels <- domestic_df %>%
  filter(!is.na(pcOth1_pct1_lag1))
domestic_future_rer_levels <- domestic_current_rer_levels %>%
  filter(
    !is.na(ln_inv_rer_code1_tplus1)
  )
domestic_current_plus_future_rer_levels <- domestic_future_rer_levels
foreign_current_rer_levels <- foreign_df %>%
  filter(!is.na(pcOth1_pct1_lag1))

domestic_current_rer_fd <- build_fd_frame(
  domestic_current_rer_levels,
  keep_cols = c("pcOth1_pct1_lag1", "pcOth1_code1")
)
domestic_future_rer_fd <- build_fd_future_rer_frame(domestic_current_rer_levels)
domestic_current_plus_future_rer_fd <- domestic_future_rer_fd
foreign_current_rer_fd <- build_fd_frame(
  foreign_current_rer_levels,
  keep_cols = c("pcOth1_pct1_lag1", "pcOth1_code1")
)

sample_counts <- bind_rows(
  sample_summary(domestic_current_rer_levels, "domestic_current_rer_levels"),
  sample_summary(domestic_future_rer_levels, "domestic_future_rer_levels"),
  sample_summary(domestic_current_plus_future_rer_levels, "domestic_current_plus_future_rer_levels"),
  sample_summary(foreign_current_rer_levels, "foreign_current_rer_levels"),
  sample_summary(domestic_current_rer_fd, "domestic_current_rer_fd"),
  sample_summary(domestic_future_rer_fd, "domestic_future_rer_fd"),
  sample_summary(domestic_current_plus_future_rer_fd, "domestic_current_plus_future_rer_fd"),
  sample_summary(foreign_current_rer_fd, "foreign_current_rer_fd")
)
write.csv(
  sample_counts,
  file.path(out_dir, "cost_reg_placebo_sample_counts.csv"),
  row.names = FALSE
)

future_rer_missing_levels <- domestic_current_rer_levels %>%
  filter(is.na(ln_inv_rer_code1_tplus1))
future_rer_missing_by_year <- future_rer_missing_levels %>%
  count(year, name = "rows")

mL_dom_current <- feols(
  ln_costs ~ ln_inv_rer_code1:pcOth1_pct1_lag1 +
    ln_size + ln_weight + ln_hp + ln_mpg | make_model + year,
  data = domestic_current_rer_levels,
  cluster = ~ make_model
)

mL_dom_future <- feols(
  ln_costs ~ ln_inv_rer_code1_tplus1:pcOth1_pct1_lag1 +
    ln_size + ln_weight + ln_hp + ln_mpg | make_model + year,
  data = domestic_future_rer_levels,
  cluster = ~ make_model
)

mL_dom_current_future <- feols(
  ln_costs ~ ln_inv_rer_code1:pcOth1_pct1_lag1 +
    ln_inv_rer_code1_tplus1:pcOth1_pct1_lag1 +
    ln_size + ln_weight + ln_hp + ln_mpg | make_model + year,
  data = domestic_current_plus_future_rer_levels,
  cluster = ~ make_model
)

mL_foreign_current <- feols(
  ln_costs ~ ln_inv_rer_code1:pcOth1_pct1_lag1 +
    ln_size + ln_weight + ln_hp + ln_mpg | make_model + year,
  data = foreign_current_rer_levels,
  cluster = ~ make_model
)

mFD_dom_current <- feols(
  ln_costs ~ ln_inv_rer_code1:pcOth1_pct1_lag1 +
    ln_size + ln_weight + ln_hp + ln_mpg | year,
  data = domestic_current_rer_fd,
  cluster = ~ make_model
)

mFD_dom_future <- feols(
  ln_costs ~ d_ln_inv_rer_code1_tplus1:pcOth1_pct1_lag1 +
    ln_size + ln_weight + ln_hp + ln_mpg | year,
  data = domestic_future_rer_fd,
  cluster = ~ make_model
)

mFD_dom_current_future <- feols(
  ln_costs ~ ln_inv_rer_code1:pcOth1_pct1_lag1 +
    d_ln_inv_rer_code1_tplus1:pcOth1_pct1_lag1 +
    ln_size + ln_weight + ln_hp + ln_mpg | year,
  data = domestic_current_plus_future_rer_fd,
  cluster = ~ make_model
)

mFD_foreign_current <- feols(
  ln_costs ~ ln_inv_rer_code1:pcOth1_pct1_lag1 +
    ln_size + ln_weight + ln_hp + ln_mpg | year,
  data = foreign_current_rer_fd,
  cluster = ~ make_model
)

levels_dict <- c(
  "ln_inv_rer_code1:pcOth1_pct1_lag1" = "$\\rho_{j,t-1}\\times\\log(RER_{jt})$",
  "pcOth1_pct1_lag1:ln_inv_rer_code1" = "$\\rho_{j,t-1}\\times\\log(RER_{jt})$",
  "ln_inv_rer_code1_tplus1:pcOth1_pct1_lag1" = "$\\rho_{j,t-1}\\times\\log(RER_{j,t+1})$",
  "pcOth1_pct1_lag1:ln_inv_rer_code1_tplus1" = "$\\rho_{j,t-1}\\times\\log(RER_{j,t+1})$",
  "ln_size" = "$\\ln(\\text{size})$",
  "ln_weight" = "$\\ln(\\text{weight})$",
  "ln_hp" = "$\\ln(\\text{hp})$",
  "ln_mpg" = "$\\ln(\\text{mpg})$"
)

fd_dict <- c(
  "ln_inv_rer_code1:pcOth1_pct1_lag1" = "$\\rho_{j,t-1}\\times\\Delta\\log(RER_{jt})$",
  "pcOth1_pct1_lag1:ln_inv_rer_code1" = "$\\rho_{j,t-1}\\times\\Delta\\log(RER_{jt})$",
  "d_ln_inv_rer_code1_tplus1:pcOth1_pct1_lag1" = "$\\rho_{j,t-1}\\times\\Delta\\log(RER_{j,t+1})$",
  "pcOth1_pct1_lag1:d_ln_inv_rer_code1_tplus1" = "$\\rho_{j,t-1}\\times\\Delta\\log(RER_{j,t+1})$",
  "ln_size" = "$\\Delta\\ln(\\text{size})$",
  "ln_weight" = "$\\Delta\\ln(\\text{weight})$",
  "ln_hp" = "$\\Delta\\ln(\\text{hp})$",
  "ln_mpg" = "$\\Delta\\ln(\\text{mpg})$"
)

levels_path <- file.path(out_dir, "cost_reg_placebo_levels_table.tex")
fd_path <- file.path(out_dir, "cost_reg_placebo_fd_table.tex")

etable(
  mL_dom_current, mL_dom_future, mL_dom_current_future, mL_foreign_current,
  tex = TRUE,
  file = levels_path,
  replace = TRUE,
  title = "Exchange-rate placebo regressions (levels)",
  label = "tab:cost_reg_placebo_levels",
  headers = list(
    "Domestic sample" = 3,
    "Foreign placebo" = 1
  ),
  dict = levels_dict,
  fitstat = ~ n + r2 + wr2,
  signif.code = c("***" = 0.01, "**" = 0.05, "*" = 0.10)
)

etable(
  mFD_dom_current, mFD_dom_future, mFD_dom_current_future, mFD_foreign_current,
  tex = TRUE,
  file = fd_path,
  replace = TRUE,
  title = "Exchange-rate placebo regressions (first differences)",
  label = "tab:cost_reg_placebo_fd",
  headers = list(
    "Domestic sample" = 3,
    "Foreign placebo" = 1
  ),
  dict = fd_dict,
  fitstat = ~ n + r2 + wr2,
  signif.code = c("***" = 0.01, "**" = 0.05, "*" = 0.10)
)

relabel_model_numbers(levels_path, c("(1)", "(2)", "(3)", "(4)"))
relabel_model_numbers(fd_path, c("(1)", "(2)", "(3)", "(4)"))

append_table_note(
  levels_path,
  "The dependent variable is log recovered marginal cost. Columns (1)--(3) use U.S.-assembled vehicles: column (1) includes the contemporaneous exchange-rate exposure term, column (2) replaces it with the one-year-ahead term, and column (3) includes both terms. Column (4) is a foreign-assembled placebo estimated with the contemporaneous exposure term on a different sample; it tests whether the same exposure construction predicts costs where the domestic imported-parts channel should not apply. All specifications include make-model and year fixed effects and vehicle controls. Standard errors are clustered by make-model."
)
append_table_note(
  fd_path,
  "The dependent variable is the first difference of log recovered marginal cost. Columns (1)--(3) use U.S.-assembled vehicles observed in consecutive years: column (1) includes the contemporaneous exchange-rate exposure term, column (2) replaces it with the one-year-ahead exchange-rate difference, and column (3) includes both terms. Column (4) is a foreign-assembled placebo estimated on a different first-difference sample. All specifications include year fixed effects and differenced vehicle controls. Standard errors are clustered by make-model."
)

coef_rows <- bind_rows(
  extract_term_row(mL_dom_current, "ln_inv_rer_code1:pcOth1_pct1_lag1", "domestic_current_rer_levels", "levels"),
  extract_term_row(mL_dom_future, "ln_inv_rer_code1_tplus1:pcOth1_pct1_lag1", "domestic_future_rer_levels", "levels"),
  extract_term_row(mL_dom_current_future, "ln_inv_rer_code1:pcOth1_pct1_lag1", "domestic_current_plus_future_rer_levels", "levels"),
  extract_term_row(mL_dom_current_future, "ln_inv_rer_code1_tplus1:pcOth1_pct1_lag1", "domestic_current_plus_future_rer_levels", "levels"),
  extract_term_row(mL_foreign_current, "ln_inv_rer_code1:pcOth1_pct1_lag1", "foreign_current_rer_levels", "levels"),
  extract_term_row(mFD_dom_current, "ln_inv_rer_code1:pcOth1_pct1_lag1", "domestic_current_rer_fd", "fd"),
  extract_term_row(mFD_dom_future, "d_ln_inv_rer_code1_tplus1:pcOth1_pct1_lag1", "domestic_future_rer_fd", "fd"),
  extract_term_row(mFD_dom_current_future, "ln_inv_rer_code1:pcOth1_pct1_lag1", "domestic_current_plus_future_rer_fd", "fd"),
  extract_term_row(mFD_dom_current_future, "d_ln_inv_rer_code1_tplus1:pcOth1_pct1_lag1", "domestic_current_plus_future_rer_fd", "fd"),
  extract_term_row(mFD_foreign_current, "ln_inv_rer_code1:pcOth1_pct1_lag1", "foreign_current_rer_fd", "fd")
)

write.csv(
  coef_rows,
  file.path(out_dir, "cost_reg_placebo_coefficients.csv"),
  row.names = FALSE
)

notes_lines <- c(
  "# Cost-Reg Placebo Notes",
  "",
  "This file is generated by `cost_side/cost_reg_placebo.R`.",
  "",
  "## Sample Definitions",
  "",
  sprintf(
    "- `domestic_current_rer_levels`: %d canonical domestic rows with lagged imported-parts exposure available.",
    nrow(domestic_current_rer_levels)
  ),
  sprintf(
    "- `domestic_future_rer_levels`: %d of those rows with a direct country-year `t+1` exchange rate from the current row's `pcOth1_code1` and `year + 1`.",
    nrow(domestic_future_rer_levels)
  ),
  sprintf(
    "- `domestic_future_rer_fd`: %d rows after differencing the direct country-year future term `log(RER_{t+1}) - log(RER_t)` on the current first-difference sample.",
    nrow(domestic_future_rer_fd)
  ),
  sprintf(
    "- `foreign_current_rer_levels`: %d foreign-assembled source-stable rows with lagged imported-parts exposure available.",
    nrow(foreign_current_rer_levels)
  ),
  "",
  "## Direct Future-RER Availability",
  "",
  sprintf(
    "- Direct future `RER_{t+1}` is available for %d of %d domestic current-sample rows.",
    nrow(domestic_future_rer_levels),
    nrow(domestic_current_rer_levels)
  ),
  sprintf(
    "- Rows without a direct country-year `t+1` exchange rate: %d.",
    nrow(future_rer_missing_levels)
  ),
  "- Missing direct future `RER_{t+1}` by current year:",
  if (nrow(future_rer_missing_by_year) == 0) {
    "- none"
  } else {
    paste0("- ", future_rer_missing_by_year$year, ": ", future_rer_missing_by_year$rows, " rows")
  },
  "",
  "## Specifications",
  "",
  "- Levels regressions match the baseline cost-side control set with make-model and year fixed effects.",
  "- The domestic levels placebo uses a direct country-year `RER_{t+1}` merge, not the next observed make-model row.",
  "- The joint domestic levels specification includes both the current and one-year-ahead exchange-rate terms so the future-shock term is evaluated conditional on the current-shock term.",
  "- The joint current-plus-future specifications are the main timing diagnostics because they test whether `rho_{t-1} x log(RER_{t+1})` predicts current recovered costs after controlling for `rho_{t-1} x log(RER_t)`.",
  "- First-difference regressions use direct country-year future differences, `log(RER_{t+1}) - log(RER_t)`, from the current row's source country.",
  "- The future first-difference placebo no longer requires a next observed make-model row; after the 2025 exchange-rate backfill it should match the current first-difference sample whenever all current-row source countries have `t+1` exchange-rate data.",
  "- The foreign placebo uses the same `pcOth1` source-country exposure construction as the domestic sample.",
  "",
  "## Output Files",
  "",
  "- `cost_reg_placebo_levels_table.tex`",
  "- `cost_reg_placebo_fd_table.tex`",
  "- `cost_reg_placebo_coefficients.csv`",
  "- `cost_reg_placebo_sample_counts.csv`"
)
writeLines(notes_lines, file.path(out_dir, "cost_reg_placebo_notes.md"))

cat("Saved:\n")
cat(" -", levels_path, "\n")
cat(" -", fd_path, "\n")
cat(" -", file.path(out_dir, "cost_reg_placebo_coefficients.csv"), "\n")
cat(" -", file.path(out_dir, "cost_reg_placebo_sample_counts.csv"), "\n")
cat(" -", file.path(out_dir, "cost_reg_placebo_notes.md"), "\n")
