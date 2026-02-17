library(dplyr)
library(fixest)

panel_path <- if (file.exists("cost_side_panel_dropped.csv")) {
  "cost_side_panel_dropped.csv"
} else {
  "cost_side/cost_side_panel_dropped.csv"
}

out_dir <- "cost_side/outputs"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

df <- read.csv(panel_path) %>%
  mutate(
    make_model = as.character(make_model),
    year = as.integer(year),
    across(
      c(costs, rer_pcOth1_code1_n2015, pcOth1_pct1, size, weight, hp, mpg),
      ~ suppressWarnings(as.numeric(.))
    )
  ) %>%
  arrange(make_model, year) %>%
  group_by(make_model) %>%
  mutate(pcOth1_pct1_lag1 = lag(pcOth1_pct1, 1)) %>%
  ungroup() %>%
  filter(costs > 0, rer_pcOth1_code1_n2015 > 0) %>%
  filter(
    !is.na(pcOth1_pct1_lag1),
    !is.na(size), !is.na(weight), !is.na(hp), !is.na(mpg)
  )

elas_path <- "post_est/data/derived/product_year_elasticities.csv"
elas <- read.csv(elas_path) %>%
  transmute(
    product_ids = as.character(product_ids),
    market_year = as.integer(market_year),
    own_elas_t = suppressWarnings(as.numeric(own_elasticity))
  )

df <- read.csv(panel_path) %>%
  mutate(
    product_ids = as.character(product_ids),
    market_year = as.integer(market_year),
    make_model = as.character(make_model),
    year = as.integer(year),
    across(
      c(costs, rer_pcOth1_code1_n2015, pcOth1_pct1, size, weight, hp, mpg),
      ~ suppressWarnings(as.numeric(.))
    )
  ) %>%
  left_join(elas, by = c("product_ids", "market_year")) %>%
  arrange(make_model, year) %>%
  group_by(make_model) %>%
  mutate(
    pcOth1_pct1_lag1 = lag(pcOth1_pct1, 1),
    log_abs_own_elas_t = log(abs(own_elas_t)),
    ln_costs = log(costs),
    ln_inv_rer_code1 = -log(rer_pcOth1_code1_n2015),
    ln_size = log(size),
    ln_weight = log(weight),
    ln_hp = log(hp),
    ln_mpg = log(mpg)
  ) %>%
  ungroup() %>%
  filter(costs > 0, rer_pcOth1_code1_n2015 > 0) %>%
  filter(
    !is.na(pcOth1_pct1_lag1),
    !is.na(size), !is.na(weight), !is.na(hp), !is.na(mpg),
    !is.na(own_elas_t), is.finite(log_abs_own_elas_t)
  )

# Robustness model with rho * log(abs(elasticity)) * log(RER)
m <- feols(
  ln_costs ~ ln_inv_rer_code1:pcOth1_pct1_lag1 +
    ln_inv_rer_code1:own_elas_t +
    ln_inv_rer_code1:pcOth1_pct1_lag1:log_abs_own_elas_t +
    ln_size + ln_weight + ln_hp + ln_mpg | make_model + year,
  data = df,
  cluster = ~ make_model
)

coefs <- coef(m)
beta_rho <- unname(coefs["ln_inv_rer_code1:pcOth1_pct1_lag1"])
beta_rho_logelas <- unname(coefs["ln_inv_rer_code1:pcOth1_pct1_lag1:log_abs_own_elas_t"])

if (!is.finite(beta_rho) || !is.finite(beta_rho_logelas)) {
  stop("Could not extract required coefficients from robustness model.")
}

abs_elas <- abs(df$own_elas_t)
abs_elas <- abs_elas[is.finite(abs_elas) & abs_elas > 0]
rho_vals <- quantile(df$pcOth1_pct1_lag1, probs = c(0.25, 0.50, 0.75), na.rm = TRUE)

eps_grid <- seq(
  quantile(abs_elas, 0.05, na.rm = TRUE),
  quantile(abs_elas, 0.95, na.rm = TRUE),
  length.out = 200
)

curve_df <- bind_rows(lapply(seq_along(rho_vals), function(i) {
  rho_i <- as.numeric(rho_vals[[i]])
  data.frame(
    abs_own_elasticity = eps_grid,
    rho = rho_i,
    rho_label = paste0(names(rho_vals)[i], " rho = ", sprintf("%.3f", rho_i)),
    implied_pass_through = rho_i * (beta_rho + beta_rho_logelas * log(eps_grid))
  )
}))

write.csv(
  curve_df,
  file.path(out_dir, "implied_pass_through_vs_elasticity.csv"),
  row.names = FALSE
)

meta <- data.frame(
  beta_rho = beta_rho,
  beta_rho_log_abs_elas = beta_rho_logelas,
  rho_p25 = as.numeric(rho_vals[[1]]),
  rho_p50 = as.numeric(rho_vals[[2]]),
  rho_p75 = as.numeric(rho_vals[[3]])
)
write.csv(
  meta,
  file.path(out_dir, "implied_pass_through_model_coeffs.csv"),
  row.names = FALSE
)

line_cols <- c("#1b9e77", "#d95f02", "#7570b3")

png(
  filename = file.path(out_dir, "implied_pass_through_vs_elasticity.png"),
  width = 1100,
  height = 700,
  res = 120
)
plot(
  NA, NA,
  xlim = range(curve_df$abs_own_elasticity),
  ylim = range(curve_df$implied_pass_through),
  xlab = "Absolute own-price elasticity |epsilon|",
  ylab = "Implied pass-through to ln(cost) for a 1 log-point RER shock",
  main = "Implied Pass-Through vs Elasticity\n(from log-elasticity robustness model)"
)
abline(h = 0, lty = 2, col = "gray50")
for (i in seq_along(rho_vals)) {
  s <- curve_df %>% filter(rho_label == unique(curve_df$rho_label)[i])
  lines(s$abs_own_elasticity, s$implied_pass_through, lwd = 2.5, col = line_cols[i])
}
legend(
  "topleft",
  legend = unique(curve_df$rho_label),
  col = line_cols,
  lwd = 2.5,
  bty = "n"
)
dev.off()

pdf(file.path(out_dir, "implied_pass_through_vs_elasticity.pdf"), width = 11, height = 7)
plot(
  NA, NA,
  xlim = range(curve_df$abs_own_elasticity),
  ylim = range(curve_df$implied_pass_through),
  xlab = "Absolute own-price elasticity |epsilon|",
  ylab = "Implied pass-through to ln(cost) for a 1 log-point RER shock",
  main = "Implied Pass-Through vs Elasticity\n(from log-elasticity robustness model)"
)
abline(h = 0, lty = 2, col = "gray50")
for (i in seq_along(rho_vals)) {
  s <- curve_df %>% filter(rho_label == unique(curve_df$rho_label)[i])
  lines(s$abs_own_elasticity, s$implied_pass_through, lwd = 2.5, col = line_cols[i])
}
legend(
  "topleft",
  legend = unique(curve_df$rho_label),
  col = line_cols,
  lwd = 2.5,
  bty = "n"
)
dev.off()

cat("Saved:\n")
cat(" -", file.path(out_dir, "implied_pass_through_vs_elasticity.png"), "\n")
cat(" -", file.path(out_dir, "implied_pass_through_vs_elasticity.pdf"), "\n")
cat(" -", file.path(out_dir, "implied_pass_through_vs_elasticity.csv"), "\n")
cat(" -", file.path(out_dir, "implied_pass_through_model_coeffs.csv"), "\n")
