source("cost_side/robustness_helpers.R")

out_dir <- ensure_outputs_dir()
panel <- load_cost_panel("cost_side_panel.csv")
df <- build_regression_frame(panel, include_prices = TRUE) %>%
  filter(
    !is.na(pcOth1_pct1_lag1),
    is.finite(ln_prices),
    is.finite(ln_markups)
  )

high_exposure_cutoff <- quantile(df$pcOth1_pct1_lag1, 0.75, na.rm = TRUE)
high_df <- df %>%
  filter(pcOth1_pct1_lag1 >= high_exposure_cutoff)

series_df <- high_df %>%
  group_by(year) %>%
  summarise(
    median_ln_costs = median(ln_costs, na.rm = TRUE),
    median_ln_prices = median(ln_prices, na.rm = TRUE),
    median_ln_inv_rer = median(ln_inv_rer_code1, na.rm = TRUE),
    observations = n(),
    .groups = "drop"
  ) %>%
  arrange(year)

series_df <- series_df %>%
  mutate(
    indexed_ln_costs = median_ln_costs - first(median_ln_costs),
    indexed_ln_prices = median_ln_prices - first(median_ln_prices),
    indexed_ln_inv_rer = median_ln_inv_rer - first(median_ln_inv_rer)
  )

write.csv(
  series_df,
  file.path(out_dir, "high_exposure_series_values.csv"),
  row.names = FALSE
)

png(
  filename = file.path(out_dir, "high_exposure_series_mc_price_rer.png"),
  width = 1200,
  height = 420,
  res = 120
)
par(mfrow = c(1, 3), mar = c(4, 4, 3, 1))
plot(
  series_df$year,
  series_df$indexed_ln_costs,
  type = "b",
  pch = 19,
  col = "#1b9e77",
  xlab = "Year",
  ylab = "Indexed log level",
  main = "Recovered marginal cost"
)
abline(h = 0, lty = 2, col = "gray60")
plot(
  series_df$year,
  series_df$indexed_ln_prices,
  type = "b",
  pch = 19,
  col = "#377eb8",
  xlab = "Year",
  ylab = "Indexed log level",
  main = "Observed price"
)
abline(h = 0, lty = 2, col = "gray60")
plot(
  series_df$year,
  series_df$indexed_ln_inv_rer,
  type = "b",
  pch = 19,
  col = "#d95f02",
  xlab = "Year",
  ylab = "Indexed log level",
  main = "Primary-source RER"
)
abline(h = 0, lty = 2, col = "gray60")
dev.off()

fd <- high_df %>%
  arrange(make_model, year) %>%
  group_by(make_model) %>%
  mutate(
    year_gap = year - lag(year),
    d_ln_costs = ln_costs - lag(ln_costs),
    d_ln_prices = ln_prices - lag(ln_prices),
    d_ln_markups = ln_markups - lag(ln_markups),
    d_ln_inv_rer = ln_inv_rer_code1 - lag(ln_inv_rer_code1),
    d_ln_size = ln_size - lag(ln_size),
    d_ln_weight = ln_weight - lag(ln_weight),
    d_ln_hp = ln_hp - lag(ln_hp),
    d_ln_mpg = ln_mpg - lag(ln_mpg)
  ) %>%
  ungroup() %>%
  filter(
    year_gap == 1,
    !is.na(d_ln_costs),
    !is.na(d_ln_prices),
    !is.na(d_ln_markups),
    !is.na(d_ln_inv_rer)
  )

resid_x_model <- feols(
  d_ln_inv_rer ~ d_ln_size + d_ln_weight + d_ln_hp + d_ln_mpg | year,
  data = fd
)
resid_cost_model <- feols(
  d_ln_costs ~ d_ln_size + d_ln_weight + d_ln_hp + d_ln_mpg | year,
  data = fd
)
resid_price_model <- feols(
  d_ln_prices ~ d_ln_size + d_ln_weight + d_ln_hp + d_ln_mpg | year,
  data = fd
)
resid_markup_model <- feols(
  d_ln_markups ~ d_ln_size + d_ln_weight + d_ln_hp + d_ln_mpg | year,
  data = fd
)

fd <- fd %>%
  mutate(
    resid_d_ln_inv_rer = resid(resid_x_model),
    resid_d_ln_costs = resid(resid_cost_model),
    resid_d_ln_prices = resid(resid_price_model),
    resid_d_ln_markups = resid(resid_markup_model)
  )

make_binned <- function(data, outcome_col, outcome_label, bins = 8L) {
  breaks <- unique(quantile(data$resid_d_ln_inv_rer, probs = seq(0, 1, length.out = bins + 1L), na.rm = TRUE))
  if (length(breaks) < 3L) {
    return(data.frame())
  }
  outcome_sym <- rlang::sym(outcome_col)
  data %>%
    mutate(bin = cut(resid_d_ln_inv_rer, breaks = breaks, include.lowest = TRUE, labels = FALSE)) %>%
    filter(!is.na(bin)) %>%
    group_by(bin) %>%
    summarise(
      outcome = outcome_label,
      mean_resid_d_ln_inv_rer = mean(resid_d_ln_inv_rer, na.rm = TRUE),
      mean_resid_outcome = mean(!!outcome_sym, na.rm = TRUE),
      n = n(),
      .groups = "drop"
    )
}

binned <- bind_rows(
  make_binned(fd, "resid_d_ln_costs", "Recovered marginal cost"),
  make_binned(fd, "resid_d_ln_prices", "Observed price"),
  make_binned(fd, "resid_d_ln_markups", "Recovered markup")
)

write.csv(
  binned,
  file.path(out_dir, "high_exposure_fd_binned.csv"),
  row.names = FALSE
)

panel_cols <- c(
  "Recovered marginal cost" = "#1b9e77",
  "Observed price" = "#377eb8",
  "Recovered markup" = "#d95f02"
)

png(
  filename = file.path(out_dir, "high_exposure_fd_decomp.png"),
  width = 1200,
  height = 420,
  res = 120
)
par(mfrow = c(1, 3), mar = c(4, 4, 3, 1))
for (outcome_label in names(panel_cols)) {
  plot_df <- binned %>%
    filter(outcome == outcome_label) %>%
    arrange(mean_resid_d_ln_inv_rer)
  plot(
    plot_df$mean_resid_d_ln_inv_rer,
    plot_df$mean_resid_outcome,
    type = "b",
    pch = 19,
    col = panel_cols[[outcome_label]],
    xlab = "Residualized delta log(RER)",
    ylab = "Residualized outcome",
    main = outcome_label
  )
  abline(h = 0, v = 0, lty = 2, col = "gray60")
}
dev.off()

notes_lines <- c(
  "# High-Exposure Visual Notes",
  "",
  "This file is generated by `cost_side/plot_high_exposure_series.R`.",
  "",
  "## Sample",
  "",
  sprintf(
    "- High exposure is defined as lagged primary foreign share in the top quartile of the domestic baseline sample (cutoff %.4f).",
    high_exposure_cutoff
  ),
  sprintf(
    "- The high-exposure levels sample contains %d rows and %d make-models.",
    nrow(high_df),
    dplyr::n_distinct(high_df$make_model)
  ),
  sprintf(
    "- The high-exposure first-difference sample contains %d rows and %d make-models.",
    nrow(fd),
    dplyr::n_distinct(fd$make_model)
  ),
  "",
  "## Output Files",
  "",
  "- `high_exposure_series_values.csv`",
  "- `high_exposure_series_mc_price_rer.png`",
  "- `high_exposure_fd_binned.csv`",
  "- `high_exposure_fd_decomp.png`"
)
writeLines(notes_lines, file.path(out_dir, "high_exposure_plot_notes.md"))

cat("Saved:\n")
cat(" -", file.path(out_dir, "high_exposure_series_values.csv"), "\n")
cat(" -", file.path(out_dir, "high_exposure_series_mc_price_rer.png"), "\n")
cat(" -", file.path(out_dir, "high_exposure_fd_binned.csv"), "\n")
cat(" -", file.path(out_dir, "high_exposure_fd_decomp.png"), "\n")
cat(" -", file.path(out_dir, "high_exposure_plot_notes.md"), "\n")
