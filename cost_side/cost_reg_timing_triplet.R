source("cost_side/robustness_helpers.R")

attach_direct_shifted_rer <- function(
  df,
  exchange_lookup,
  shift_years,
  code_col = "pcOth1_code1",
  year_col = "year",
  out_col
) {
  keys <- data.frame(
    .row_id = seq_len(nrow(df)),
    country = as.character(df[[code_col]]),
    year = as.integer(df[[year_col]]) + shift_years,
    stringsAsFactors = FALSE
  )

  matched <- keys %>%
    left_join(exchange_lookup, by = c("country", "year")) %>%
    arrange(.row_id)

  out <- df
  out[[out_col]] <- matched$ln_inv_rer
  out
}

build_levels_triplet_frame <- function(df, exchange_lookup) {
  df %>%
    attach_direct_shifted_rer(
      exchange_lookup = exchange_lookup,
      shift_years = -1L,
      out_col = "ln_inv_rer_code1_lag1"
    ) %>%
    filter(!is.na(ln_inv_rer_code1_lag1), !is.na(ln_inv_rer_code1_tplus1))
}

build_fd_triplet_frame <- function(df, exchange_lookup) {
  df %>%
    attach_direct_shifted_rer(
      exchange_lookup = exchange_lookup,
      shift_years = -1L,
      out_col = "ln_inv_rer_code1_lag1"
    ) %>%
    attach_direct_shifted_rer(
      exchange_lookup = exchange_lookup,
      shift_years = -2L,
      out_col = "ln_inv_rer_code1_lag2"
    ) %>%
    arrange(make_model, year) %>%
    group_by(make_model) %>%
    mutate(
      year_gap = year - lag(year),
      d_ln_costs = ln_costs - lag(ln_costs),
      d_ln_inv_rer_code1 = ln_inv_rer_code1 - lag(ln_inv_rer_code1),
      d_ln_inv_rer_code1_lag1 = ln_inv_rer_code1_lag1 - ln_inv_rer_code1_lag2,
      d_ln_inv_rer_code1_tplus1 = ln_inv_rer_code1_tplus1 - ln_inv_rer_code1,
      d_ln_size = ln_size - lag(ln_size),
      d_ln_weight = ln_weight - lag(ln_weight),
      d_ln_hp = ln_hp - lag(ln_hp),
      d_ln_mpg = ln_mpg - lag(ln_mpg)
    ) %>%
    ungroup() %>%
    filter(
      year_gap == 1,
      !is.na(d_ln_costs),
      !is.na(d_ln_inv_rer_code1),
      !is.na(d_ln_inv_rer_code1_lag1),
      !is.na(d_ln_inv_rer_code1_tplus1)
    ) %>%
    transmute(
      make_model = make_model,
      year = year,
      pcOth1_code1 = pcOth1_code1,
      ln_costs = d_ln_costs,
      ln_inv_rer_code1 = d_ln_inv_rer_code1,
      d_ln_inv_rer_code1_lag1 = d_ln_inv_rer_code1_lag1,
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

domestic_df <- build_regression_frame(domestic_panel) %>%
  attach_direct_future_rer(
    exchange_lookup,
    code_col = "pcOth1_code1",
    year_col = "year",
    out_col = "ln_inv_rer_code1_tplus1"
  ) %>%
  filter(!is.na(pcOth1_pct1_lag1))

levels_triplet_df <- build_levels_triplet_frame(domestic_df, exchange_lookup)
fd_triplet_df <- build_fd_triplet_frame(domestic_df, exchange_lookup)

sample_counts <- bind_rows(
  sample_summary(levels_triplet_df, "levels_triplet_sample"),
  sample_summary(fd_triplet_df, "fd_triplet_sample")
)
write.csv(
  sample_counts,
  file.path(out_dir, "cost_reg_timing_triplet_sample_counts.csv"),
  row.names = FALSE
)

mL_current_future <- feols(
  ln_costs ~ ln_inv_rer_code1:pcOth1_pct1_lag1 +
    ln_inv_rer_code1_tplus1:pcOth1_pct1_lag1 +
    ln_size + ln_weight + ln_hp + ln_mpg | make_model + year,
  data = levels_triplet_df,
  cluster = ~ make_model
)

mL_lag_current_future <- feols(
  ln_costs ~ ln_inv_rer_code1_lag1:pcOth1_pct1_lag1 +
    ln_inv_rer_code1:pcOth1_pct1_lag1 +
    ln_inv_rer_code1_tplus1:pcOth1_pct1_lag1 +
    ln_size + ln_weight + ln_hp + ln_mpg | make_model + year,
  data = levels_triplet_df,
  cluster = ~ make_model
)

mL_lag_current <- feols(
  ln_costs ~ ln_inv_rer_code1_lag1:pcOth1_pct1_lag1 +
    ln_inv_rer_code1:pcOth1_pct1_lag1 +
    ln_size + ln_weight + ln_hp + ln_mpg | make_model + year,
  data = levels_triplet_df,
  cluster = ~ make_model
)

mFD_current_future <- feols(
  ln_costs ~ ln_inv_rer_code1:pcOth1_pct1_lag1 +
    d_ln_inv_rer_code1_tplus1:pcOth1_pct1_lag1 +
    ln_size + ln_weight + ln_hp + ln_mpg | year,
  data = fd_triplet_df,
  cluster = ~ make_model
)

mFD_lag_current <- feols(
  ln_costs ~ d_ln_inv_rer_code1_lag1:pcOth1_pct1_lag1 +
    ln_inv_rer_code1:pcOth1_pct1_lag1 +
    ln_size + ln_weight + ln_hp + ln_mpg | year,
  data = fd_triplet_df,
  cluster = ~ make_model
)

mFD_lag_current_future <- feols(
  ln_costs ~ d_ln_inv_rer_code1_lag1:pcOth1_pct1_lag1 +
    ln_inv_rer_code1:pcOth1_pct1_lag1 +
    d_ln_inv_rer_code1_tplus1:pcOth1_pct1_lag1 +
    ln_size + ln_weight + ln_hp + ln_mpg | year,
  data = fd_triplet_df,
  cluster = ~ make_model
)

levels_dict <- c(
  "ln_inv_rer_code1_lag1:pcOth1_pct1_lag1" = "$\\rho_{j,t-1}\\times\\log(RER_{j,t-1})$",
  "pcOth1_pct1_lag1:ln_inv_rer_code1_lag1" = "$\\rho_{j,t-1}\\times\\log(RER_{j,t-1})$",
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
  "d_ln_inv_rer_code1_lag1:pcOth1_pct1_lag1" = "$\\rho_{j,t-1}\\times\\Delta\\log(RER_{j,t-1})$",
  "pcOth1_pct1_lag1:d_ln_inv_rer_code1_lag1" = "$\\rho_{j,t-1}\\times\\Delta\\log(RER_{j,t-1})$",
  "ln_inv_rer_code1:pcOth1_pct1_lag1" = "$\\rho_{j,t-1}\\times\\Delta\\log(RER_{jt})$",
  "pcOth1_pct1_lag1:ln_inv_rer_code1" = "$\\rho_{j,t-1}\\times\\Delta\\log(RER_{jt})$",
  "d_ln_inv_rer_code1_tplus1:pcOth1_pct1_lag1" = "$\\rho_{j,t-1}\\times\\Delta\\log(RER_{j,t+1})$",
  "pcOth1_pct1_lag1:d_ln_inv_rer_code1_tplus1" = "$\\rho_{j,t-1}\\times\\Delta\\log(RER_{j,t+1})$",
  "ln_size" = "$\\Delta\\ln(\\text{size})$",
  "ln_weight" = "$\\Delta\\ln(\\text{weight})$",
  "ln_hp" = "$\\Delta\\ln(\\text{hp})$",
  "ln_mpg" = "$\\Delta\\ln(\\text{mpg})$"
)

levels_path <- file.path(out_dir, "cost_reg_timing_triplet_levels_table.tex")
fd_path <- file.path(out_dir, "cost_reg_timing_triplet_fd_table.tex")

etable(
  mL_current_future, mL_lag_current_future, mL_lag_current,
  tex = TRUE,
  file = levels_path,
  replace = TRUE,
  title = "Current, future, and lagged exchange-rate timing regressions (levels)",
  label = "tab:cost_reg_timing_triplet_levels",
  headers = list(
    "Current + future" = 1,
    "Lag + current + future" = 1,
    "Lag + current" = 1
  ),
  dict = levels_dict,
  fitstat = ~ n + r2 + wr2,
  signif.code = c("***" = 0.01, "**" = 0.05, "*" = 0.10)
)

etable(
  mFD_current_future, mFD_lag_current_future, mFD_lag_current,
  tex = TRUE,
  file = fd_path,
  replace = TRUE,
  title = "Current, future, and lagged exchange-rate timing regressions (first differences)",
  label = "tab:cost_reg_timing_triplet_fd",
  headers = list(
    "Current + future" = 1,
    "Lag + current + future" = 1,
    "Lag + current" = 1
  ),
  dict = fd_dict,
  fitstat = ~ n + r2 + wr2,
  signif.code = c("***" = 0.01, "**" = 0.05, "*" = 0.10)
)

relabel_model_numbers(levels_path, c("(1)", "(2)", "(3)"))
relabel_model_numbers(fd_path, c("(1)", "(2)", "(3)"))

append_table_note(
  levels_path,
  "The dependent variable is log recovered marginal cost. All columns use the same levels timing sample with non-missing lagged, contemporaneous, and one-year-ahead source-country exchange rates. Column (1) includes contemporaneous and one-year-ahead exposure terms. Column (2) adds the lagged exposure term. Column (3) includes lagged and contemporaneous exposure terms and excludes the one-year-ahead term. All specifications include make-model and year fixed effects and vehicle controls. Standard errors are clustered by make-model."
)
append_table_note(
  fd_path,
  "The dependent variable is the first difference of log recovered marginal cost. All columns use the same consecutive-year timing sample with non-missing lagged, contemporaneous, and one-year-ahead exchange-rate differences. Column (1) includes contemporaneous and one-year-ahead exposure terms. Column (2) adds the lagged exposure term. Column (3) includes lagged and contemporaneous exposure terms and excludes the one-year-ahead term. All specifications include year fixed effects and differenced vehicle controls. Standard errors are clustered by make-model."
)

coef_rows <- bind_rows(
  extract_term_row(mL_current_future, "ln_inv_rer_code1:pcOth1_pct1_lag1", "levels_current_future", "levels"),
  extract_term_row(mL_current_future, "ln_inv_rer_code1_tplus1:pcOth1_pct1_lag1", "levels_current_future", "levels"),
  extract_term_row(mL_lag_current_future, "ln_inv_rer_code1_lag1:pcOth1_pct1_lag1", "levels_lag_current_future", "levels"),
  extract_term_row(mL_lag_current_future, "ln_inv_rer_code1:pcOth1_pct1_lag1", "levels_lag_current_future", "levels"),
  extract_term_row(mL_lag_current_future, "ln_inv_rer_code1_tplus1:pcOth1_pct1_lag1", "levels_lag_current_future", "levels"),
  extract_term_row(mL_lag_current, "ln_inv_rer_code1_lag1:pcOth1_pct1_lag1", "levels_lag_current", "levels"),
  extract_term_row(mL_lag_current, "ln_inv_rer_code1:pcOth1_pct1_lag1", "levels_lag_current", "levels"),
  extract_term_row(mFD_current_future, "ln_inv_rer_code1:pcOth1_pct1_lag1", "fd_current_future", "fd"),
  extract_term_row(mFD_current_future, "d_ln_inv_rer_code1_tplus1:pcOth1_pct1_lag1", "fd_current_future", "fd"),
  extract_term_row(mFD_lag_current_future, "d_ln_inv_rer_code1_lag1:pcOth1_pct1_lag1", "fd_lag_current_future", "fd"),
  extract_term_row(mFD_lag_current_future, "ln_inv_rer_code1:pcOth1_pct1_lag1", "fd_lag_current_future", "fd"),
  extract_term_row(mFD_lag_current_future, "d_ln_inv_rer_code1_tplus1:pcOth1_pct1_lag1", "fd_lag_current_future", "fd"),
  extract_term_row(mFD_lag_current, "d_ln_inv_rer_code1_lag1:pcOth1_pct1_lag1", "fd_lag_current", "fd"),
  extract_term_row(mFD_lag_current, "ln_inv_rer_code1:pcOth1_pct1_lag1", "fd_lag_current", "fd")
)

write.csv(
  coef_rows,
  file.path(out_dir, "cost_reg_timing_triplet_coefficients.csv"),
  row.names = FALSE
)

notes_lines <- c(
  "# Timing Triplet Robustness",
  "",
  "This file is generated by `cost_side/cost_reg_timing_triplet.R`.",
  "",
  "## Purpose",
  "",
  "- This script compares the existing current-plus-future timing regressions to a lag-plus-current-plus-future timing regression.",
  "- The goal is to test whether the `t+1` term is just proxying for exchange-rate persistence that should instead load on `t-1`.",
  "",
  "## Sample Construction",
  "",
  sprintf(
    "- Levels triplet sample: %d rows with non-missing `RER_{t-1}`, `RER_t`, and direct country-year `RER_{t+1}`.",
    nrow(levels_triplet_df)
  ),
  sprintf(
    "- First-difference triplet sample: %d rows with non-missing `\\Delta RER_{t-1}`, `\\Delta RER_t`, and direct country-year `\\Delta RER_{t+1}`.",
    nrow(fd_triplet_df)
  ),
  "- Column (1) uses only the current and future timing terms on the matched triplet sample.",
  "- Column (2) adds the lagged timing term.",
  "- Column (3) includes the lagged and current timing terms but excludes the future timing term.",
  "",
  "## Fixed Effects",
  "",
  "- Levels: `make_model + year`.",
  "- First differences: `year`.",
  "",
  "## Output Files",
  "",
  "- `cost_reg_timing_triplet_levels_table.tex`",
  "- `cost_reg_timing_triplet_fd_table.tex`",
  "- `cost_reg_timing_triplet_coefficients.csv`",
  "- `cost_reg_timing_triplet_sample_counts.csv`"
)
writeLines(notes_lines, file.path(out_dir, "cost_reg_timing_triplet_notes.md"))

cat("Saved:\n")
cat(" -", levels_path, "\n")
cat(" -", fd_path, "\n")
cat(" -", file.path(out_dir, "cost_reg_timing_triplet_coefficients.csv"), "\n")
cat(" -", file.path(out_dir, "cost_reg_timing_triplet_sample_counts.csv"), "\n")
cat(" -", file.path(out_dir, "cost_reg_timing_triplet_notes.md"), "\n")
