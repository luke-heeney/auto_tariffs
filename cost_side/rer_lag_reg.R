# ============================================================
# Regress lagged exchange rate on import share (code1 country)
# ============================================================

# install.packages(c("dplyr", "fixest"))

library(dplyr)
library(fixest)

final_df <- read.csv("cost_side_panel.csv")

# Use code1 bilateral RER when available
ex_rate_col <- if ("rer_pcOth1_code1_n2015" %in% names(final_df)) {
  "rer_pcOth1_code1_n2015"
} else {
  stop("Exchange-rate column rer_pcOth1_code1_n2015 not found.")
}

# Build analysis frame
df <- final_df %>%
  transmute(
    rer_code1 = .data[[ex_rate_col]],
    pcOth1_pct1,
    make_model = as.character(make_model),
    year = as.integer(year)
  ) %>%
  mutate(
    rer_code1 = suppressWarnings(as.numeric(rer_code1)),
    pcOth1_pct1 = suppressWarnings(as.numeric(pcOth1_pct1))
  ) %>%
  arrange(make_model, year) %>%
  group_by(make_model) %>%
  mutate(
    ln_rer_code1 = log(rer_code1),
    ln_rer_code1_lag1 = dplyr::lag(ln_rer_code1, 1)
  ) %>%
  ungroup() %>%
  filter(rer_code1 > 0) %>%                # remove non-positive FX values before log
  filter(!is.na(pcOth1_pct1), !is.na(ln_rer_code1_lag1))

cat("N rows used:", nrow(df), "\n")
cat("N make_model clusters:", dplyr::n_distinct(df$make_model), "\n\n")

print(summary(df$pcOth1_pct1))

# Regressions: import share on lagged exchange rate (with/without year FE)
m_share <- feols(
  pcOth1_pct1 ~ ln_rer_code1_lag1 | make_model,
  data = df,
  cluster = ~ make_model
)

m_share_year <- feols(
  pcOth1_pct1 ~ ln_rer_code1_lag1 | make_model + year,
  data = df,
  cluster = ~ make_model
)

etable(
  m_share, m_share_year,
  keep = c("ln_rer_code1_lag1"),
  se.below = TRUE
)

# Coefficient of interest only
list(
  no_year = coeftable(m_share)["ln_rer_code1_lag1", , drop = FALSE],
  with_year = coeftable(m_share_year)["ln_rer_code1_lag1", , drop = FALSE]
)
