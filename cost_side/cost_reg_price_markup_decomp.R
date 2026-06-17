source("cost_side/robustness_helpers.R")

build_fd_outcome_frame <- function(df, outcome_col, keep_cols = character()) {
  outcome_sym <- rlang::sym(outcome_col)
  df %>%
    arrange(make_model, year) %>%
    group_by(make_model) %>%
    mutate(
      year_gap = year - lag(year),
      d_y = !!outcome_sym - lag(!!outcome_sym),
      d_ln_inv_rer_code1 = ln_inv_rer_code1 - lag(ln_inv_rer_code1),
      d_ln_size = ln_size - lag(ln_size),
      d_ln_weight = ln_weight - lag(ln_weight),
      d_ln_hp = ln_hp - lag(ln_hp),
      d_ln_mpg = ln_mpg - lag(ln_mpg)
    ) %>%
    ungroup() %>%
    filter(year_gap == 1, !is.na(d_y), !is.na(d_ln_inv_rer_code1)) %>%
    transmute(
      make_model = make_model,
      year = year,
      y = d_y,
      ln_inv_rer_code1 = d_ln_inv_rer_code1,
      !!!rlang::syms(keep_cols),
      ln_size = d_ln_size,
      ln_weight = d_ln_weight,
      ln_hp = d_ln_hp,
      ln_mpg = d_ln_mpg
    )
}

build_fd_future_outcome_frame <- function(df, outcome_col, keep_cols = character()) {
  outcome_sym <- rlang::sym(outcome_col)
  df %>%
    arrange(make_model, year) %>%
    group_by(make_model) %>%
    mutate(
      year_gap = year - lag(year),
      d_y = !!outcome_sym - lag(!!outcome_sym),
      d_ln_inv_rer_code1_tplus1 = ln_inv_rer_code1_tplus1 - ln_inv_rer_code1,
      d_ln_size = ln_size - lag(ln_size),
      d_ln_weight = ln_weight - lag(ln_weight),
      d_ln_hp = ln_hp - lag(ln_hp),
      d_ln_mpg = ln_mpg - lag(ln_mpg)
    ) %>%
    ungroup() %>%
    filter(year_gap == 1, !is.na(d_y), !is.na(d_ln_inv_rer_code1_tplus1)) %>%
    transmute(
      make_model = make_model,
      year = year,
      y = d_y,
      d_ln_inv_rer_code1_tplus1 = d_ln_inv_rer_code1_tplus1,
      !!!rlang::syms(keep_cols),
      ln_size = d_ln_size,
      ln_weight = d_ln_weight,
      ln_hp = d_ln_hp,
      ln_mpg = d_ln_mpg
    )
}

run_levels_model <- function(df, outcome_col, rer_term) {
  feols(
    as.formula(
      paste0(
        outcome_col,
        " ~ ",
        rer_term,
        ":pcOth1_pct1_lag1 + ln_size + ln_weight + ln_hp + ln_mpg | make_model + year"
      )
    ),
    data = df,
    cluster = ~ make_model
  )
}

run_fd_model <- function(df, rer_term) {
  feols(
    as.formula(
      paste0(
        "y ~ ",
        rer_term,
        ":pcOth1_pct1_lag1 + ln_size + ln_weight + ln_hp + ln_mpg | year"
      )
    ),
    data = df,
    cluster = ~ make_model
  )
}

out_dir <- ensure_outputs_dir()
exchange_lookup <- load_normalized_exchange_lookup()
panel <- load_cost_panel("cost_side_panel.csv")
df <- build_regression_frame(panel, include_prices = TRUE) %>%
  filter(
    !is.na(pcOth1_pct1_lag1),
    is.finite(ln_markups),
    is.finite(ln_prices)
  ) %>%
  attach_direct_future_rer(
    exchange_lookup,
    code_col = "pcOth1_code1",
    year_col = "year",
    out_col = "ln_inv_rer_code1_tplus1"
  )

current_levels <- df
future_levels <- df %>% filter(!is.na(ln_inv_rer_code1_tplus1))

outcomes <- c("ln_costs", "ln_prices", "ln_markups")
outcome_labels <- c(
  recovered_cost = "Recovered cost",
  observed_price = "Observed price",
  recovered_markup = "Recovered markup"
)

current_level_models <- setNames(
  lapply(outcomes, function(outcome_col) run_levels_model(current_levels, outcome_col, "ln_inv_rer_code1")),
  outcomes
)
future_level_models <- setNames(
  lapply(outcomes, function(outcome_col) run_levels_model(future_levels, outcome_col, "ln_inv_rer_code1_tplus1")),
  outcomes
)

current_fd_frames <- setNames(
  lapply(outcomes, function(outcome_col) {
    build_fd_outcome_frame(current_levels, outcome_col, keep_cols = c("pcOth1_pct1_lag1"))
  }),
  outcomes
)
future_fd_frames <- setNames(
  lapply(outcomes, function(outcome_col) {
    build_fd_future_outcome_frame(future_levels, outcome_col, keep_cols = c("pcOth1_pct1_lag1"))
  }),
  outcomes
)

current_fd_models <- setNames(
  lapply(current_fd_frames, run_fd_model, rer_term = "ln_inv_rer_code1"),
  outcomes
)
future_fd_models <- setNames(
  lapply(future_fd_frames, run_fd_model, rer_term = "d_ln_inv_rer_code1_tplus1"),
  outcomes
)

dict_current <- c(
  "ln_inv_rer_code1:pcOth1_pct1_lag1" = "Current RER exposure term",
  "pcOth1_pct1_lag1:ln_inv_rer_code1" = "Current RER exposure term",
  "ln_inv_rer_code1_tplus1:pcOth1_pct1_lag1" = "Forward RER exposure term",
  "pcOth1_pct1_lag1:ln_inv_rer_code1_tplus1" = "Forward RER exposure term",
  "d_ln_inv_rer_code1_tplus1:pcOth1_pct1_lag1" = "Forward RER exposure term",
  "pcOth1_pct1_lag1:d_ln_inv_rer_code1_tplus1" = "Forward RER exposure term",
  "ln_size" = "$\\ln(\\text{size})$",
  "ln_weight" = "$\\ln(\\text{weight})$",
  "ln_hp" = "$\\ln(\\text{hp})$",
  "ln_mpg" = "$\\ln(\\text{mpg})$"
)

baseline_table_path <- file.path(out_dir, "cost_reg_price_markup_decomp_table.tex")
forward_table_path <- file.path(out_dir, "cost_reg_price_markup_forward_placebo_table.tex")

etable(
  current_level_models$ln_costs,
  current_level_models$ln_prices,
  current_level_models$ln_markups,
  current_fd_models$ln_costs,
  current_fd_models$ln_prices,
  current_fd_models$ln_markups,
  tex = TRUE,
  file = baseline_table_path,
  replace = TRUE,
  title = "Baseline exchange-rate decomposition across recovered costs, prices, and markups",
  label = "tab:cost_reg_price_markup_decomp",
  headers = list(
    "Levels" = 3,
    "First differences" = 3
  ),
  dict = dict_current,
  fitstat = ~ n + r2 + wr2,
  signif.code = c("***" = 0.01, "**" = 0.05, "*" = 0.10)
)

etable(
  future_level_models$ln_costs,
  future_level_models$ln_prices,
  future_level_models$ln_markups,
  future_fd_models$ln_costs,
  future_fd_models$ln_prices,
  future_fd_models$ln_markups,
  tex = TRUE,
  file = forward_table_path,
  replace = TRUE,
  title = "Forward-RER placebo decomposition across recovered costs, prices, and markups",
  label = "tab:cost_reg_price_markup_forward_placebo",
  headers = list(
    "Levels" = 3,
    "First differences" = 3
  ),
  dict = dict_current,
  fitstat = ~ n + r2 + wr2,
  signif.code = c("***" = 0.01, "**" = 0.05, "*" = 0.10)
)

relabel_model_numbers(
  baseline_table_path,
  c("(1)", "(2)", "(3)", "(4)", "(5)", "(6)")
)
relabel_model_numbers(
  forward_table_path,
  c("(1)", "(2)", "(3)", "(4)", "(5)", "(6)")
)

append_table_note(
  baseline_table_path,
  "Columns (1)--(3) are levels regressions with make-model and year fixed effects; columns (4)--(6) are first-difference regressions with year fixed effects. The dependent variables are log recovered marginal cost, log observed price, and log recovered markup, respectively. The reported exchange-rate coefficient is the contemporaneous source-country exchange-rate term interacted with lagged imported-parts exposure. All regressions include the vehicle controls shown in the table. Standard errors are clustered by make-model."
)
append_table_note(
  forward_table_path,
  "Columns (1)--(3) are levels regressions with make-model and year fixed effects; columns (4)--(6) are first-difference regressions with year fixed effects. The dependent variables are log recovered marginal cost, log observed price, and log recovered markup, respectively. The reported exchange-rate coefficient is the one-year-ahead source-country exchange-rate term interacted with lagged imported-parts exposure. All regressions include the vehicle controls shown in the table. Standard errors are clustered by make-model."
)

coef_rows <- bind_rows(
  extract_term_row(current_level_models$ln_costs, "ln_inv_rer_code1:pcOth1_pct1_lag1", "recovered_cost", "levels_current"),
  extract_term_row(current_level_models$ln_prices, "ln_inv_rer_code1:pcOth1_pct1_lag1", "observed_price", "levels_current"),
  extract_term_row(current_level_models$ln_markups, "ln_inv_rer_code1:pcOth1_pct1_lag1", "recovered_markup", "levels_current"),
  extract_term_row(current_fd_models$ln_costs, "ln_inv_rer_code1:pcOth1_pct1_lag1", "recovered_cost", "fd_current"),
  extract_term_row(current_fd_models$ln_prices, "ln_inv_rer_code1:pcOth1_pct1_lag1", "observed_price", "fd_current"),
  extract_term_row(current_fd_models$ln_markups, "ln_inv_rer_code1:pcOth1_pct1_lag1", "recovered_markup", "fd_current"),
  extract_term_row(future_level_models$ln_costs, "ln_inv_rer_code1_tplus1:pcOth1_pct1_lag1", "recovered_cost", "levels_forward"),
  extract_term_row(future_level_models$ln_prices, "ln_inv_rer_code1_tplus1:pcOth1_pct1_lag1", "observed_price", "levels_forward"),
  extract_term_row(future_level_models$ln_markups, "ln_inv_rer_code1_tplus1:pcOth1_pct1_lag1", "recovered_markup", "levels_forward"),
  extract_term_row(future_fd_models$ln_costs, "d_ln_inv_rer_code1_tplus1:pcOth1_pct1_lag1", "recovered_cost", "fd_forward"),
  extract_term_row(future_fd_models$ln_prices, "d_ln_inv_rer_code1_tplus1:pcOth1_pct1_lag1", "observed_price", "fd_forward"),
  extract_term_row(future_fd_models$ln_markups, "d_ln_inv_rer_code1_tplus1:pcOth1_pct1_lag1", "recovered_markup", "fd_forward")
) %>%
  mutate(outcome_label = unname(outcome_labels[sample]))

write.csv(
  coef_rows,
  file.path(out_dir, "cost_reg_price_markup_decomp_coefficients.csv"),
  row.names = FALSE
)

notes_lines <- c(
  "# Price/Markup Decomposition Notes",
  "",
  "This file is generated by `cost_side/cost_reg_price_markup_decomp.R`.",
  "",
  "## Construction",
  "",
  "- Start from the canonical domestic sample in `cost_side_panel.csv`.",
  "- Merge observed vehicle prices from `post_est/data/raw/blpUS0804.csv` onto the product-year panel.",
  "- Compare the exchange-rate exposure coefficient across three dependent variables: recovered log marginal cost, observed log price, and recovered log markup.",
  "- Repeat the same decomposition for the direct forward-RER placebo specification.",
  "",
  "## Output Files",
  "",
  "- `cost_reg_price_markup_decomp_table.tex`",
  "- `cost_reg_price_markup_forward_placebo_table.tex`",
  "- `cost_reg_price_markup_decomp_coefficients.csv`"
)
writeLines(notes_lines, file.path(out_dir, "cost_reg_price_markup_decomp_notes.md"))

cat("Saved:\n")
cat(" -", baseline_table_path, "\n")
cat(" -", forward_table_path, "\n")
cat(" -", file.path(out_dir, "cost_reg_price_markup_decomp_coefficients.csv"), "\n")
cat(" -", file.path(out_dir, "cost_reg_price_markup_decomp_notes.md"), "\n")
