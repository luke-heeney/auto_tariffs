source("cost_side/robustness_helpers.R")

build_fd_alt_exposure_frame <- function(df, keep_cols = character()) {
  df %>%
    arrange(make_model, year) %>%
    group_by(make_model) %>%
    mutate(
      year_gap = year - lag(year),
      d_ln_costs = ln_costs - lag(ln_costs),
      d_ln_inv_rer_code1 = ln_inv_rer_code1 - lag(ln_inv_rer_code1),
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
      !is.na(d_ln_inv_rer_code1_tplus1)
    ) %>%
    transmute(
      make_model = make_model,
      year = year,
      ln_costs = d_ln_costs,
      ln_inv_rer_code1 = d_ln_inv_rer_code1,
      d_ln_inv_rer_code1_tplus1 = d_ln_inv_rer_code1_tplus1,
      !!!rlang::syms(keep_cols),
      ln_size = d_ln_size,
      ln_weight = d_ln_weight,
      ln_hp = d_ln_hp,
      ln_mpg = d_ln_mpg
    )
}

out_dir <- ensure_outputs_dir()
exchange_lookup <- load_normalized_exchange_lookup()
panel <- load_cost_panel("cost_side_panel.csv")
df <- build_regression_frame(panel) %>%
  filter(!is.na(pcOth1_pct1_lag1), !is.na(observed_foreign_share_lag1)) %>%
  attach_direct_future_rer(
    exchange_lookup,
    code_col = "pcOth1_code1",
    year_col = "year",
    out_col = "ln_inv_rer_code1_tplus1"
  )

high_exposure_cutoff <- quantile(df$pcOth1_pct1_lag1, 0.75, na.rm = TRUE)
df <- df %>%
  mutate(
    high_exposure_q4_lag1 = as.integer(pcOth1_pct1_lag1 >= high_exposure_cutoff)
  )

fd <- build_fd_alt_exposure_frame(
  df,
  keep_cols = c("pcOth1_pct1_lag1", "observed_foreign_share_lag1", "high_exposure_q4_lag1")
)

m_primary_current <- feols(
  ln_costs ~ ln_inv_rer_code1:pcOth1_pct1_lag1 +
    ln_size + ln_weight + ln_hp + ln_mpg | year,
  data = fd,
  cluster = ~ make_model
)

m_observed_current <- feols(
  ln_costs ~ ln_inv_rer_code1:observed_foreign_share_lag1 +
    ln_size + ln_weight + ln_hp + ln_mpg | year,
  data = fd,
  cluster = ~ make_model
)

m_high_current <- feols(
  ln_costs ~ ln_inv_rer_code1 +
    ln_inv_rer_code1:high_exposure_q4_lag1 +
    ln_size + ln_weight + ln_hp + ln_mpg | year,
  data = fd,
  cluster = ~ make_model
)

m_primary_forward <- feols(
  ln_costs ~ d_ln_inv_rer_code1_tplus1:pcOth1_pct1_lag1 +
    ln_size + ln_weight + ln_hp + ln_mpg | year,
  data = fd,
  cluster = ~ make_model
)

m_observed_forward <- feols(
  ln_costs ~ d_ln_inv_rer_code1_tplus1:observed_foreign_share_lag1 +
    ln_size + ln_weight + ln_hp + ln_mpg | year,
  data = fd,
  cluster = ~ make_model
)

m_high_forward <- feols(
  ln_costs ~ d_ln_inv_rer_code1_tplus1 +
    d_ln_inv_rer_code1_tplus1:high_exposure_q4_lag1 +
    ln_size + ln_weight + ln_hp + ln_mpg | year,
  data = fd,
  cluster = ~ make_model
)

dict <- c(
  "ln_inv_rer_code1" = "$\\Delta\\log(RER_{jt})$",
  "d_ln_inv_rer_code1_tplus1" = "$\\Delta\\log(RER_{j,t+1})$",
  "ln_inv_rer_code1:pcOth1_pct1_lag1" = "Primary-share exposure term",
  "pcOth1_pct1_lag1:ln_inv_rer_code1" = "Primary-share exposure term",
  "ln_inv_rer_code1:observed_foreign_share_lag1" = "Observed-total exposure term",
  "observed_foreign_share_lag1:ln_inv_rer_code1" = "Observed-total exposure term",
  "ln_inv_rer_code1:high_exposure_q4_lag1" = "Top-quartile exposure interaction",
  "high_exposure_q4_lag1:ln_inv_rer_code1" = "Top-quartile exposure interaction",
  "d_ln_inv_rer_code1_tplus1:pcOth1_pct1_lag1" = "Primary-share exposure term",
  "pcOth1_pct1_lag1:d_ln_inv_rer_code1_tplus1" = "Primary-share exposure term",
  "d_ln_inv_rer_code1_tplus1:observed_foreign_share_lag1" = "Observed-total exposure term",
  "observed_foreign_share_lag1:d_ln_inv_rer_code1_tplus1" = "Observed-total exposure term",
  "d_ln_inv_rer_code1_tplus1:high_exposure_q4_lag1" = "Top-quartile exposure interaction",
  "high_exposure_q4_lag1:d_ln_inv_rer_code1_tplus1" = "Top-quartile exposure interaction",
  "ln_size" = "$\\Delta\\ln(\\text{size})$",
  "ln_weight" = "$\\Delta\\ln(\\text{weight})$",
  "ln_hp" = "$\\Delta\\ln(\\text{hp})$",
  "ln_mpg" = "$\\Delta\\ln(\\text{mpg})$"
)

table_path <- file.path(out_dir, "cost_reg_alt_exposure_table.tex")

etable(
  m_primary_current,
  m_observed_current,
  m_high_current,
  m_primary_forward,
  m_observed_forward,
  m_high_forward,
  tex = TRUE,
  file = table_path,
  replace = TRUE,
  title = "Alternative exposure definitions in first-difference cost regressions",
  label = "tab:cost_reg_alt_exposure",
  headers = list(
    "Current RER" = 3,
    "Forward RER" = 3
  ),
  dict = dict,
  fitstat = ~ n + r2 + wr2,
  signif.code = c("***" = 0.01, "**" = 0.05, "*" = 0.10)
)

relabel_model_numbers(
  table_path,
  c("(1)", "(2)", "(3)", "(4)", "(5)", "(6)")
)

append_table_note(
  table_path,
  "The dependent variable is the first difference of log recovered marginal cost. Columns (1)--(3) use contemporaneous exchange-rate differences; columns (4)--(6) use one-year-ahead exchange-rate differences. Columns (1) and (4) use the lagged primary-source foreign-parts share. Columns (2) and (5) use the lagged sum of the first two observed foreign-parts shares. Columns (3) and (6) use an indicator for lagged primary-source exposure in the top quartile and include the unrestricted exchange-rate main effect. All specifications include year fixed effects and differenced vehicle controls. Standard errors are clustered by make-model."
)

coef_rows <- bind_rows(
  extract_term_row(m_primary_current, "ln_inv_rer_code1:pcOth1_pct1_lag1", "primary_share", "fd_current"),
  extract_term_row(m_observed_current, "ln_inv_rer_code1:observed_foreign_share_lag1", "observed_total_share", "fd_current"),
  extract_term_row(m_high_current, "ln_inv_rer_code1", "high_exposure_indicator", "fd_current"),
  extract_term_row(m_high_current, "ln_inv_rer_code1:high_exposure_q4_lag1", "high_exposure_indicator", "fd_current"),
  extract_term_row(m_primary_forward, "d_ln_inv_rer_code1_tplus1:pcOth1_pct1_lag1", "primary_share", "fd_forward"),
  extract_term_row(m_observed_forward, "d_ln_inv_rer_code1_tplus1:observed_foreign_share_lag1", "observed_total_share", "fd_forward"),
  extract_term_row(m_high_forward, "d_ln_inv_rer_code1_tplus1", "high_exposure_indicator", "fd_forward"),
  extract_term_row(m_high_forward, "d_ln_inv_rer_code1_tplus1:high_exposure_q4_lag1", "high_exposure_indicator", "fd_forward")
)

write.csv(
  coef_rows,
  file.path(out_dir, "cost_reg_alt_exposure_coefficients.csv"),
  row.names = FALSE
)

notes_lines <- c(
  "# Alternative Exposure Notes",
  "",
  "This file is generated by `cost_side/cost_reg_alt_exposure.R`.",
  "",
  "## Exposure Definitions",
  "",
  "- `primary_share`: lagged `pcOth1_pct1`, the baseline primary-source foreign share.",
  "- `observed_total_share`: lagged `pcOth1_pct1 + pcOth1_pct2`, the total observed foreign share across the first two foreign slots.",
  sprintf(
    "- `high_exposure_indicator`: an indicator for lagged primary foreign share in the top quartile of the domestic baseline sample (cutoff %.4f).",
    high_exposure_cutoff
  ),
  "",
  "## Scope",
  "",
  "- This robustness table uses only the preferred first-difference designs.",
  "- Columns (1)-(3) use current exchange-rate differences; columns (4)-(6) use forward exchange-rate differences.",
  "",
  "## Output Files",
  "",
  "- `cost_reg_alt_exposure_table.tex`",
  "- `cost_reg_alt_exposure_coefficients.csv`"
)
writeLines(notes_lines, file.path(out_dir, "cost_reg_alt_exposure_notes.md"))

cat("Saved:\n")
cat(" -", table_path, "\n")
cat(" -", file.path(out_dir, "cost_reg_alt_exposure_coefficients.csv"), "\n")
cat(" -", file.path(out_dir, "cost_reg_alt_exposure_notes.md"), "\n")
