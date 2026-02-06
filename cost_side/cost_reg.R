# ============================================================
# R version with diagnostics + fixes (recommended: fixest)
# ============================================================

# install.packages(c("dplyr","fixest","broom"))
# For wild cluster bootstrap (recommended if few clusters):
# install.packages("fwildclusterboot")

library(dplyr)
library(fixest)
data_path <- if (file.exists('cost_side_panel.csv')) 'cost_side_panel.csv' else 'cost_side/cost_side_panel.csv'
final_df <- read.csv(data_path)
# ---- 0) Pick the index column defensively ----
index_col <- if ("rer_index" %in% names(final_df)) "rer_index" else "tw_rer_index"

# ---- 1) Build estimation df (mirrors your Python logic) ----
df <- final_df %>%
  transmute(
    costs,
    rer_pcOth1_code1_n2015,
    rer_index = .data[[index_col]],
    pcOth1_pct1,
    make_model = as.character(make_model),
    make = as.character(make),
    year = as.integer(year),
    size, weight, hp, mpg
  ) %>%
  mutate(across(c(costs, rer_pcOth1_code1_n2015, rer_index, pcOth1_pct1,
                  size, weight, hp, mpg), ~ suppressWarnings(as.numeric(.)))) %>%
  arrange(make_model, year) %>%
  group_by(make_model) %>%
  mutate(pcOth1_pct1_lag1 = dplyr::lag(pcOth1_pct1, 1)) %>%
  ungroup() %>%
  filter(costs > 0, rer_pcOth1_code1_n2015 > 0, rer_index > 0) %>%
  filter(!is.na(pcOth1_pct1_lag1), !is.na(size), !is.na(weight), !is.na(hp), !is.na(mpg)) %>%
  mutate(
    ln_costs          = log(costs),
    ln_inv_rer_code1  = -log(rer_pcOth1_code1_n2015), # ln(1/RER)
    ln_inv_rer_index  = -log(rer_index),             # ln(1/index)
    ln_size   = log(size),
    ln_weight = log(weight),
    ln_hp     = log(hp),
    ln_mpg    = log(mpg)
  )

# ---- 2) Quick diagnostics you should always print ----
cat("N rows:", nrow(df), "\n")
cat("N make_model clusters:", dplyr::n_distinct(df$make_model), "\n")
cat("Years:", paste(range(df$year, na.rm = TRUE), collapse = "–"), "\n\n")

print(summary(df$pcOth1_pct1_lag1))
cat("\nShare of zero exposure (lag1):",
    mean(df$pcOth1_pct1_lag1 == 0, na.rm = TRUE), "\n\n")

# Optional: check within-make_model variation in exposure
within_sd <- df %>%
  group_by(make_model) %>%
  summarise(sd_exposure = sd(pcOth1_pct1_lag1, na.rm = TRUE), .groups = "drop")
cat("Median within-model SD of exposure:", median(within_sd$sd_exposure, na.rm = TRUE), "\n\n")

# ============================================================
# 3) Regressions
#   Key fixes:
#   - include ln_inv_rer_index (you computed it but forgot to include in Python)
#   - include main effects via * (so interaction + both main effects)
#   - use high-dim FE estimation (no explicit dummy explosion)
# ============================================================

# (A) Your intended “shift-share” style but WITH the missing macro FX control
m_shiftshare <- feols(
  ln_costs ~ ln_inv_rer_code1:pcOth1_pct1_lag1 + ln_inv_rer_index +
    ln_size + ln_weight + ln_hp + ln_mpg | make_model + year,
  data = df,
  cluster = ~ make_model
)

# (A0) Lean shift-share: interaction only, NO characteristics
m_shiftshare_lean <- feols(
  ln_costs ~ ln_inv_rer_code1:pcOth1_pct1_lag1 + ln_inv_rer_index | make_model + year,
  data = df,
  cluster = ~ make_model
)
# Note: ln_inv_rer_index will still be dropped if it's time-only (collinear with year FE).

# (A1) Lean shift-share with make FEs
m_shiftshare_make_lean <- feols(
  ln_costs ~ ln_inv_rer_code1:pcOth1_pct1_lag1 | make + year,
  data = df,
  cluster = ~ make_model
)


# (B) Less restrictive: include main effects + interaction using *
#     This is usually what you want unless you truly believe “only-through-exposure.”
m_full <- feols(
  ln_costs ~ ln_inv_rer_code1:pcOth1_pct1_lag1 + ln_inv_rer_code1 +
    ln_size + ln_weight + ln_hp + ln_mpg | make_model + year,
  data = df,
  cluster = ~ make_model
)

# (C) Sensitivity: drop characteristics that are nearly time-invariant within make_model
#     (often reduces collinearity / instability)
m_lean <- feols(
  ln_costs ~ ln_inv_rer_code1 * pcOth1_pct1_lag1 + ln_inv_rer_index | make_model + year,
  data = df,
  cluster = ~ make_model
)

# ---- 4) Print results (focus on coefficient of interest) ----
etable(
  m_shiftshare, m_shiftshare_lean, m_full, m_lean,
  keep = c("ln_inv_rer_code1:pcOth1_pct1_lag1", "ln_inv_rer_index",
           "ln_inv_rer_code1", "pcOth1_pct1_lag1"),
  se.below = TRUE
)


# If you want just the exact coefficient-of-interest line:
coeftable(m_full)["ln_inv_rer_code1:pcOth1_pct1_lag1", , drop = FALSE]

# ============================================================
# 5) If you have few clusters: Wild cluster bootstrap p-values
# ============================================================

# install.packages("fwildclusterboot")
# library(fwildclusterboot)
#
# # Wild cluster bootstrap for the interaction term in m_full
# wb <- boottest(
#   m_full,
#   clustid = "make_model",
#   param = "ln_inv_rer_code1:pcOth1_pct1_lag1",
#   B = 9999,            # increase for publication-grade p-values
#   bootstrap_type = "wild",
#   impose_null = TRUE
# )
# print(wb)

library(dplyr)
library(fixest)

# ============================================================
# CHECK 1: Does ln_inv_rer_code1 vary within year?
# (If it's basically constant within year, year FE would kill it.)
# ============================================================

within_year_var <- df %>%
  group_by(year) %>%
  summarise(
    n = n(),
    n_unique = n_distinct(ln_inv_rer_code1),
    sd = sd(ln_inv_rer_code1, na.rm = TRUE),
    p10 = quantile(ln_inv_rer_code1, 0.10, na.rm = TRUE),
    p50 = quantile(ln_inv_rer_code1, 0.50, na.rm = TRUE),
    p90 = quantile(ln_inv_rer_code1, 0.90, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(p90_p10 = p90 - p10)

print(within_year_var)

cat("\nYears where ln_inv_rer_code1 is constant within year (n_unique == 1):\n")
print(within_year_var %>% filter(n_unique == 1) %>% select(year, n, n_unique, sd))


# Extra: How much variation is within-year vs between-year?
# (Not perfect decomposition, but a helpful scalar.)
overall_sd <- sd(df$ln_inv_rer_code1, na.rm = TRUE)
mean_within_year_sd <- mean(within_year_var$sd, na.rm = TRUE)

cat("\nOverall SD of ln_inv_rer_code1:", overall_sd, "\n")
cat("Mean within-year SD:", mean_within_year_sd, "\n")
cat("Within/Overall ratio:", mean_within_year_sd / overall_sd, "\n")


# ============================================================
# CHECK 2: Implied marginal effect of ln_inv_rer_code1 on ln_costs
# Under m_full: d ln_cost / d ln_inv_rer_code1 = b1 + b3 * exposure
# ============================================================

b1 <- coef(m_full)["ln_inv_rer_code1"]
b3 <- coef(m_full)["ln_inv_rer_code1:pcOth1_pct1_lag1"]

# If any coefficient got dropped, stop early with a clear message
if (is.na(b1) | is.na(b3)) {
  stop("m_full does not contain both ln_inv_rer_code1 and the interaction; check the model formula.")
}

# Marginal effect at key points in the exposure distribution
qs <- c(0.10, 0.25, 0.50, 0.75, 0.90)
s_q <- quantile(df$pcOth1_pct1_lag1, qs, na.rm = TRUE)

me_q <- b1 + b3 * s_q
out_tab <- data.frame(
  quantile = qs,
  exposure = as.numeric(s_q),
  marginal_effect = as.numeric(me_q)
)
print(out_tab)

# A simple curve over the observed exposure range
s_grid <- seq(min(df$pcOth1_pct1_lag1, na.rm = TRUE),
              max(df$pcOth1_pct1_lag1, na.rm = TRUE),
              length.out = 200)

me_grid <- b1 + b3 * s_grid

plot(s_grid, me_grid, type = "l",
     xlab = "Exposure: pcOth1_pct1_lag1",
     ylab = "Marginal effect: d ln(costs) / d ln(inv_rer_code1)",
     main = "Implied marginal effect vs exposure")

abline(h = 0, lty = 2)

# Mark the median exposure
s_med <- median(df$pcOth1_pct1_lag1, na.rm = TRUE)
points(s_med, b1 + b3 * s_med, pch = 19)
text(s_med, b1 + b3 * s_med, labels = " median", pos = 4)


# ============================================================
# OPTIONAL (nice): Cluster-robust CI band for the marginal effect curve
# Uses vcov from fixest.
# ============================================================

V <- vcov(m_full, cluster = "make_model")

# Names (should exist)
n1 <- "ln_inv_rer_code1"
n3 <- "ln_inv_rer_code1:pcOth1_pct1_lag1"

if (all(c(n1, n3) %in% rownames(V))) {
  var_b1 <- V[n1, n1]
  var_b3 <- V[n3, n3]
  cov_b1b3 <- V[n1, n3]
  
  se_me <- sqrt(var_b1 + (s_grid^2) * var_b3 + 2 * s_grid * cov_b1b3)
  
  upper <- me_grid + 1.96 * se_me
  lower <- me_grid - 1.96 * se_me
  
  lines(s_grid, upper, lty = 3)
  lines(s_grid, lower, lty = 3)
  legend("topleft",
         legend = c("ME", "95% CI"),
         lty = c(1, 3), bty = "n")
}

