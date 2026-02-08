library(dplyr)
library(fixest)

data_path <- if (file.exists('cost_side_panel_dropped.csv')) 'cost_side_panel_dropped.csv' else 'cost_side/cost_side_panel_dropped.csv'
final_df <- read.csv(data_path)

# ---- Build df ----
df <- final_df %>%
  transmute(
    costs,
    rer_pcOth1_code1_n2015,
    pcOth1_pct1,
    make_model = as.character(make_model),
    year = as.integer(year),
    size, weight, hp, mpg
  ) %>%
  mutate(across(
    c(costs, rer_pcOth1_code1_n2015, pcOth1_pct1, size, weight, hp, mpg),
    ~ suppressWarnings(as.numeric(.))
  )) %>%
  arrange(make_model, year) %>%
  group_by(make_model) %>%
  mutate(pcOth1_pct1_lag1 = lag(pcOth1_pct1, 1)) %>%
  ungroup() %>%
  filter(costs > 0, rer_pcOth1_code1_n2015 > 0) %>%
  filter(
    !is.na(pcOth1_pct1_lag1),
    !is.na(size), !is.na(weight), !is.na(hp), !is.na(mpg)
  ) %>%
  mutate(
    ln_costs         = log(costs),
    ln_inv_rer_code1 = -log(rer_pcOth1_code1_n2015),
    ln_size          = log(size),
    ln_weight        = log(weight),
    ln_hp            = log(hp),
    ln_mpg           = log(mpg)
  )

# ---- Levels ----
m1_levels <- feols(
  ln_costs ~ ln_inv_rer_code1:pcOth1_pct1_lag1 +
    ln_size + ln_weight + ln_hp + ln_mpg | make_model + year,
  data = df,
  cluster = ~ make_model
)

m2_levels <- feols(
  ln_costs ~ ln_inv_rer_code1:pcOth1_pct1_lag1 | make_model + year,
  data = df,
  cluster = ~ make_model
)

# ---- First differences (consecutive years) ----
df_fd2 <- df %>%
  arrange(make_model, year) %>%
  group_by(make_model) %>%
  mutate(
    year_gap = year - lag(year),
    d_ln_costs         = ln_costs - lag(ln_costs),
    d_ln_inv_rer_code1 = ln_inv_rer_code1 - lag(ln_inv_rer_code1),
    d_ln_size          = ln_size - lag(ln_size),
    d_ln_weight        = ln_weight - lag(ln_weight),
    d_ln_hp            = ln_hp - lag(ln_hp),
    d_ln_mpg           = ln_mpg - lag(ln_mpg)
  ) %>%
  ungroup() %>%
  filter(year_gap == 1) %>%
  filter(!is.na(d_ln_costs), !is.na(d_ln_inv_rer_code1)) %>%
  transmute(
    make_model, year,
    ln_costs         = d_ln_costs,
    ln_inv_rer_code1 = d_ln_inv_rer_code1,
    pcOth1_pct1_lag1  = pcOth1_pct1_lag1,  # lagged exposure in levels
    ln_size           = d_ln_size,
    ln_weight         = d_ln_weight,
    ln_hp             = d_ln_hp,
    ln_mpg            = d_ln_mpg
  )

m1_fd <- feols(
  ln_costs ~ ln_inv_rer_code1:pcOth1_pct1_lag1 +
    ln_size + ln_weight + ln_hp + ln_mpg | year,
  data = df_fd2,
  cluster = ~ make_model
)

m2_fd <- feols(
  ln_costs ~ ln_inv_rer_code1:pcOth1_pct1_lag1 | year,
  data = df_fd2,
  cluster = ~ make_model
)

# ---- Table ----
# IMPORTANT: keep uses regex to match interaction in either order
etable(
  "Levels: chars" = m1_levels,
  "Levels: lean"  = m2_levels,
  "FD: chars"     = m1_fd,
  "FD: lean"      = m2_fd,
  se.below = TRUE
)



names(coef(m1_levels))

