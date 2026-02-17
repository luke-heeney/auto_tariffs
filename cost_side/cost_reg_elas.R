library(dplyr)
library(fixest)

# -----------------------------
# Paths
# -----------------------------
panel_path <- if (file.exists("cost_side_panel_dropped.csv")) {
  "cost_side_panel_dropped.csv"
} else {
  "cost_side/cost_side_panel_dropped.csv"
}

elas_path <- if (file.exists("post_est/data/derived/product_year_elasticities.csv")) {
  "post_est/data/derived/product_year_elasticities.csv"
} else {
  stop("Missing elasticity file: post_est/data/derived/product_year_elasticities.csv")
}

out_dir <- "cost_side/outputs"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# -----------------------------
# Load + merge
# -----------------------------
panel <- read.csv(panel_path) %>%
  mutate(
    product_ids = as.character(product_ids),
    market_year = as.integer(market_year),
    year = as.integer(year),
    assembly1 = as.character(assembly1),
    make_model = as.character(make_model)
  )

elas <- read.csv(elas_path) %>%
  transmute(
    product_ids = as.character(product_ids),
    market_year = as.integer(market_year),
    own_elas_t = suppressWarnings(as.numeric(own_elasticity))
  )

merge_diag <- panel %>%
  mutate(in_panel = 1L) %>%
  full_join(elas %>% mutate(in_elas = 1L), by = c("product_ids", "market_year")) %>%
  mutate(
    in_panel = ifelse(is.na(in_panel), 0L, in_panel),
    in_elas = ifelse(is.na(in_elas), 0L, in_elas)
  ) %>%
  summarise(
    panel_rows = sum(in_panel),
    elas_rows = sum(in_elas),
    matched_rows = sum(in_panel == 1L & in_elas == 1L),
    panel_only_rows = sum(in_panel == 1L & in_elas == 0L),
    elas_only_rows = sum(in_panel == 0L & in_elas == 1L)
  )

write.csv(merge_diag, file.path(out_dir, "cost_reg_elas_merge_diagnostics.csv"), row.names = FALSE)

df <- panel %>%
  # Domestic-only sample: US plant country
  filter(plant_country == "United States") %>%
  left_join(elas, by = c("product_ids", "market_year")) %>%
  mutate(
    across(
      c(costs, rer_pcOth1_code1_n2015, pcOth1_pct1, size, weight, hp, mpg, own_elas_t),
      ~ suppressWarnings(as.numeric(.))
    )
  ) %>%
  arrange(make_model, year) %>%
  group_by(make_model) %>%
  mutate(
    pcOth1_pct1_lag1 = lag(pcOth1_pct1, 1),
    own_elas_lag1 = lag(own_elas_t, 1)
  ) %>%
  ungroup() %>%
  filter(costs > 0, rer_pcOth1_code1_n2015 > 0) %>%
  filter(
    !is.na(pcOth1_pct1_lag1),
    !is.na(size), !is.na(weight), !is.na(hp), !is.na(mpg),
    !is.na(own_elas_t), !is.na(own_elas_lag1)
  ) %>%
  mutate(
    log_abs_own_elas_t = log(abs(own_elas_t)),
    log_abs_own_elas_lag1 = log(abs(own_elas_lag1)),
    ln_costs = log(costs),
    ln_inv_rer_code1 = -log(rer_pcOth1_code1_n2015),
    ln_size = log(size),
    ln_weight = log(weight),
    ln_hp = log(hp),
    ln_mpg = log(mpg)
  ) %>%
  filter(
    is.finite(log_abs_own_elas_t),
    is.finite(log_abs_own_elas_lag1)
  )

# -----------------------------
# Levels regressions
# Robustness focus:
#   1) Interaction structure
#   2) elas_t vs elas_{t-1}
# -----------------------------
mL_base <- feols(
  ln_costs ~ ln_inv_rer_code1:pcOth1_pct1_lag1 +
    ln_size + ln_weight + ln_hp + ln_mpg | make_model + year,
  data = df, cluster = ~ make_model
)

# two-way interaction with elas_t
mL_tw_t <- feols(
  ln_costs ~ ln_inv_rer_code1:pcOth1_pct1_lag1 +
    ln_inv_rer_code1:own_elas_t +
    ln_size + ln_weight + ln_hp + ln_mpg | make_model + year,
  data = df, cluster = ~ make_model
)

# two-way interaction with elas_{t-1}
mL_tw_l1 <- feols(
  ln_costs ~ ln_inv_rer_code1:pcOth1_pct1_lag1 +
    ln_inv_rer_code1:own_elas_lag1 +
    ln_size + ln_weight + ln_hp + ln_mpg | make_model + year,
  data = df, cluster = ~ make_model
)

# triple interaction with elas_t
mL_tri_t <- feols(
  ln_costs ~ ln_inv_rer_code1:pcOth1_pct1_lag1 +
    ln_inv_rer_code1:own_elas_t +
    ln_inv_rer_code1:pcOth1_pct1_lag1:own_elas_t +
    ln_size + ln_weight + ln_hp + ln_mpg | make_model + year,
  data = df, cluster = ~ make_model
)

# triple interaction with elas_{t-1}
mL_tri_l1 <- feols(
  ln_costs ~ ln_inv_rer_code1:pcOth1_pct1_lag1 +
    ln_inv_rer_code1:own_elas_lag1 +
    ln_inv_rer_code1:pcOth1_pct1_lag1:own_elas_lag1 +
    ln_size + ln_weight + ln_hp + ln_mpg | make_model + year,
  data = df, cluster = ~ make_model
)

# triple interaction with log(abs(elas_t))
mL_tri_log_t <- feols(
  ln_costs ~ ln_inv_rer_code1:pcOth1_pct1_lag1 +
    ln_inv_rer_code1:own_elas_t +
    ln_inv_rer_code1:pcOth1_pct1_lag1:log_abs_own_elas_t +
    ln_size + ln_weight + ln_hp + ln_mpg | make_model + year,
  data = df, cluster = ~ make_model
)

# triple interaction with log(abs(elas_{t-1}))
mL_tri_log_l1 <- feols(
  ln_costs ~ ln_inv_rer_code1:pcOth1_pct1_lag1 +
    ln_inv_rer_code1:own_elas_lag1 +
    ln_inv_rer_code1:pcOth1_pct1_lag1:log_abs_own_elas_lag1 +
    ln_size + ln_weight + ln_hp + ln_mpg | make_model + year,
  data = df, cluster = ~ make_model
)

# -----------------------------
# First differences (consecutive years)
# -----------------------------
df_fd <- df %>%
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
    d_own_elas_t = own_elas_t - lag(own_elas_t)
  ) %>%
  ungroup() %>%
  filter(year_gap == 1) %>%
  filter(!is.na(d_ln_costs), !is.na(d_ln_inv_rer_code1)) %>%
  transmute(
    make_model, year,
    ln_costs = d_ln_costs,
    ln_inv_rer_code1 = d_ln_inv_rer_code1,
    pcOth1_pct1_lag1 = pcOth1_pct1_lag1,    # level lagged exposure
    own_elas_t = own_elas_t,                # level current elas
    own_elas_lag1 = own_elas_lag1,          # level lagged elas
    log_abs_own_elas_t = log_abs_own_elas_t,
    log_abs_own_elas_lag1 = log_abs_own_elas_lag1,
    d_own_elas_t = d_own_elas_t,            # optional differenced elas
    ln_size = d_ln_size,
    ln_weight = d_ln_weight,
    ln_hp = d_ln_hp,
    ln_mpg = d_ln_mpg
  )

mFD_base <- feols(
  ln_costs ~ ln_inv_rer_code1:pcOth1_pct1_lag1 +
    ln_size + ln_weight + ln_hp + ln_mpg | year,
  data = df_fd, cluster = ~ make_model
)

mFD_tw_t <- feols(
  ln_costs ~ ln_inv_rer_code1:pcOth1_pct1_lag1 +
    ln_inv_rer_code1:own_elas_t +
    ln_size + ln_weight + ln_hp + ln_mpg | year,
  data = df_fd, cluster = ~ make_model
)

mFD_tw_l1 <- feols(
  ln_costs ~ ln_inv_rer_code1:pcOth1_pct1_lag1 +
    ln_inv_rer_code1:own_elas_lag1 +
    ln_size + ln_weight + ln_hp + ln_mpg | year,
  data = df_fd, cluster = ~ make_model
)

mFD_tri_t <- feols(
  ln_costs ~ ln_inv_rer_code1:pcOth1_pct1_lag1 +
    ln_inv_rer_code1:own_elas_t +
    ln_inv_rer_code1:pcOth1_pct1_lag1:own_elas_t +
    ln_size + ln_weight + ln_hp + ln_mpg | year,
  data = df_fd, cluster = ~ make_model
)

mFD_tri_l1 <- feols(
  ln_costs ~ ln_inv_rer_code1:pcOth1_pct1_lag1 +
    ln_inv_rer_code1:own_elas_lag1 +
    ln_inv_rer_code1:pcOth1_pct1_lag1:own_elas_lag1 +
    ln_size + ln_weight + ln_hp + ln_mpg | year,
  data = df_fd, cluster = ~ make_model
)

# triple interaction with log(abs(elas_t))
mFD_tri_log_t <- feols(
  ln_costs ~ ln_inv_rer_code1:pcOth1_pct1_lag1 +
    ln_inv_rer_code1:own_elas_t +
    ln_inv_rer_code1:pcOth1_pct1_lag1:log_abs_own_elas_t +
    ln_size + ln_weight + ln_hp + ln_mpg | year,
  data = df_fd, cluster = ~ make_model
)

# triple interaction with log(abs(elas_{t-1}))
mFD_tri_log_l1 <- feols(
  ln_costs ~ ln_inv_rer_code1:pcOth1_pct1_lag1 +
    ln_inv_rer_code1:own_elas_lag1 +
    ln_inv_rer_code1:pcOth1_pct1_lag1:log_abs_own_elas_lag1 +
    ln_size + ln_weight + ln_hp + ln_mpg | year,
  data = df_fd, cluster = ~ make_model
)

# -----------------------------
# Export LaTeX
# -----------------------------
dict <- c(
  "ln_inv_rer_code1:pcOth1_pct1_lag1" = "$\\rho_{j,t-1}\\times\\log(RER_{jt})$",
  "pcOth1_pct1_lag1:ln_inv_rer_code1" = "$\\rho_{j,t-1}\\times\\log(RER_{jt})$",
  "ln_inv_rer_code1:own_elas_t" = "$\\varepsilon_{j,t}\\times\\log(RER_{jt})$",
  "own_elas_t:ln_inv_rer_code1" = "$\\varepsilon_{j,t}\\times\\log(RER_{jt})$",
  "ln_inv_rer_code1:own_elas_lag1" = "$\\varepsilon_{j,t-1}\\times\\log(RER_{jt})$",
  "own_elas_lag1:ln_inv_rer_code1" = "$\\varepsilon_{j,t-1}\\times\\log(RER_{jt})$",
  "ln_inv_rer_code1:pcOth1_pct1_lag1:own_elas_t" = "$\\rho_{j,t-1}\\times\\varepsilon_{j,t}\\times\\log(RER_{jt})$",
  "ln_inv_rer_code1:pcOth1_pct1_lag1:own_elas_lag1" = "$\\rho_{j,t-1}\\times\\varepsilon_{j,t-1}\\times\\log(RER_{jt})$",
  "ln_inv_rer_code1:pcOth1_pct1_lag1:log_abs_own_elas_t" = "$\\rho_{j,t-1}\\times\\log|\\varepsilon_{j,t}|\\times\\log(RER_{jt})$",
  "ln_inv_rer_code1:pcOth1_pct1_lag1:log_abs_own_elas_lag1" = "$\\rho_{j,t-1}\\times\\log|\\varepsilon_{j,t-1}|\\times\\log(RER_{jt})$",
  "ln_size" = "$\\ln(\\text{size})$",
  "ln_weight" = "$\\ln(\\text{weight})$",
  "ln_hp" = "$\\ln(\\text{hp})$",
  "ln_mpg" = "$\\ln(\\text{mpg})$"
)

etable(
  mL_base, mL_tw_t, mL_tw_l1, mL_tri_t, mL_tri_l1, mL_tri_log_t, mL_tri_log_l1,
  tex = TRUE,
  file = file.path(out_dir, "cost_reg_elas_levels_table.tex"),
  replace = TRUE,
  title = "Cost-side regressions with elasticity interactions (levels, domestic vehicles)",
  label = "tab:cost_reg_elas_levels",
  dict = dict,
  fitstat = ~n + r2 + wr2,
  signif.code = c("***" = 0.01, "**" = 0.05, "*" = 0.10)
)

etable(
  mFD_base, mFD_tw_t, mFD_tw_l1, mFD_tri_t, mFD_tri_l1, mFD_tri_log_t, mFD_tri_log_l1,
  tex = TRUE,
  file = file.path(out_dir, "cost_reg_elas_fd_table.tex"),
  replace = TRUE,
  title = "Cost-side regressions with elasticity interactions (first differences, domestic vehicles)",
  label = "tab:cost_reg_elas_fd",
  dict = dict,
  fitstat = ~n + r2 + wr2,
  signif.code = c("***" = 0.01, "**" = 0.05, "*" = 0.10)
)

cat("Saved:\n")
cat(" -", file.path(out_dir, "cost_reg_elas_merge_diagnostics.csv"), "\n")
cat(" -", file.path(out_dir, "cost_reg_elas_levels_table.tex"), "\n")
cat(" -", file.path(out_dir, "cost_reg_elas_fd_table.tex"), "\n")
