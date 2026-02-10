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

# ---- Export LaTeX table ----
models <- list(m1_levels, m2_levels, m1_fd, m2_fd)

fmt_num <- function(x, digits = 3) {
  sprintf(paste0("%.", digits, "f"), x)
}

sig_star <- function(p) {
  if (is.na(p)) return("")
  if (p < 0.01) return("\\sym{***}")
  if (p < 0.05) return("\\sym{**}")
  if (p < 0.10) return("\\sym{*}")
  ""
}

coef_se_cell <- function(model, term_pattern) {
  ct <- summary(model)$coeftable
  term_name <- grep(term_pattern, rownames(ct), value = TRUE)
  if (length(term_name) == 0) {
    return(list(coef = "", se = ""))
  }
  term_name <- term_name[1]
  est <- ct[term_name, "Estimate"]
  se <- ct[term_name, "Std. Error"]
  p <- ct[term_name, "Pr(>|t|)"]
  list(
    coef = paste0("$", fmt_num(est), "$", sig_star(p)),
    se = paste0("$(", fmt_num(se), ")$")
  )
}

fit_stat <- function(model, stat) {
  as.numeric(fitstat(model, stat)[[1]])
}

row_cells <- function(pattern, use_models = c(TRUE, TRUE, TRUE, TRUE)) {
  vals <- vector("list", 4)
  for (i in seq_along(models)) {
    if (use_models[i]) {
      vals[[i]] <- coef_se_cell(models[[i]], pattern)
    } else {
      vals[[i]] <- list(coef = "", se = "")
    }
  }
  vals
}

size_cells <- row_cells("^ln_size$", c(TRUE, FALSE, TRUE, FALSE))
weight_cells <- row_cells("^ln_weight$", c(TRUE, FALSE, TRUE, FALSE))
hp_cells <- row_cells("^ln_hp$", c(TRUE, FALSE, TRUE, FALSE))
mpg_cells <- row_cells("^ln_mpg$", c(TRUE, FALSE, TRUE, FALSE))
int_cells <- row_cells("ln_inv_rer_code1:pcOth1_pct1_lag1|pcOth1_pct1_lag1:ln_inv_rer_code1")

n_vec <- sapply(models, nobs)
r2_vec <- sapply(models, fit_stat, stat = "r2")
wr2_vec <- sapply(models, fit_stat, stat = "wr2")

tex_lines <- c(
  "% Requires: \\usepackage{booktabs,threeparttable}",
  "\\begin{table}[!htbp]",
  "\\centering",
  "\\begin{threeparttable}",
  "\\caption{Cost-Side Regressions: Exchange-Rate Shocks and Imported Parts Exposure}",
  "\\label{tab:cost_side_results}",
  "\\setlength{\\tabcolsep}{5pt}",
  "\\renewcommand{\\arraystretch}{1.12}",
  "\\newcommand{\\sym}[1]{\\ifmmode^{#1}\\else\\(^{#1}\\)\\fi}",
  "\\begin{tabular}{lcccc}",
  "\\toprule",
  " & \\multicolumn{2}{c}{\\textbf{Levels}} & \\multicolumn{2}{c}{\\textbf{First-differences}} \\\\",
  "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
  " & \\textbf{(1)} & \\textbf{(2)} & \\textbf{(3)} & \\textbf{(4)} \\\\",
  "\\midrule",
  paste0("$\\ln(\\text{size})$",
         "\n  & ", size_cells[[1]]$coef, " & ", size_cells[[2]]$coef, " & ", size_cells[[3]]$coef, " & ", size_cells[[4]]$coef, " \\\\",
         "\n  & ", size_cells[[1]]$se,   " & ", size_cells[[2]]$se,   " & ", size_cells[[3]]$se,   " & ", size_cells[[4]]$se,   " \\\\"),
  paste0("$\\ln(\\text{weight})$",
         "\n  & ", weight_cells[[1]]$coef, " & ", weight_cells[[2]]$coef, " & ", weight_cells[[3]]$coef, " & ", weight_cells[[4]]$coef, " \\\\",
         "\n  & ", weight_cells[[1]]$se,   " & ", weight_cells[[2]]$se,   " & ", weight_cells[[3]]$se,   " & ", weight_cells[[4]]$se,   " \\\\"),
  paste0("$\\ln(\\text{hp})$",
         "\n  & ", hp_cells[[1]]$coef, " & ", hp_cells[[2]]$coef, " & ", hp_cells[[3]]$coef, " & ", hp_cells[[4]]$coef, " \\\\",
         "\n  & ", hp_cells[[1]]$se,   " & ", hp_cells[[2]]$se,   " & ", hp_cells[[3]]$se,   " & ", hp_cells[[4]]$se,   " \\\\"),
  paste0("$\\ln(\\text{mpg})$",
         "\n  & ", mpg_cells[[1]]$coef, " & ", mpg_cells[[2]]$coef, " & ", mpg_cells[[3]]$coef, " & ", mpg_cells[[4]]$coef, " \\\\",
         "\n  & ", mpg_cells[[1]]$se,   " & ", mpg_cells[[2]]$se,   " & ", mpg_cells[[3]]$se,   " & ", mpg_cells[[4]]$se,   " \\\\"),
  paste0("$\\rho^{(1)}_{f,j,t-1}\\cdot \\log\\!\\big(RER^{(1)}_{jt}\\big)$",
         "\n  & ", int_cells[[1]]$coef, " & ", int_cells[[2]]$coef, " & ", int_cells[[3]]$coef, " & ", int_cells[[4]]$coef, " \\\\",
         "\n  & ", int_cells[[1]]$se,   " & ", int_cells[[2]]$se,   " & ", int_cells[[3]]$se,   " & ", int_cells[[4]]$se,   " \\\\"),
  "\\midrule",
  "Make-model FE & Yes & Yes & No  & No  \\\\",
  "Year FE       & Yes & Yes & Yes & Yes \\\\",
  "\\midrule",
  paste0("Observations  & ", n_vec[1], " & ", n_vec[2], " & ", n_vec[3], " & ", n_vec[4], " \\\\"),
  paste0("$R^2$         & ", fmt_num(r2_vec[1]), " & ", fmt_num(r2_vec[2]), " & ", fmt_num(r2_vec[3]), " & ", fmt_num(r2_vec[4]), " \\\\"),
  paste0("Within $R^2$  & ", fmt_num(wr2_vec[1]), " & ", fmt_num(wr2_vec[2]), " & ", fmt_num(wr2_vec[3]), " & ", fmt_num(wr2_vec[4]), " \\\\"),
  "\\bottomrule",
  "\\end{tabular}",
  "\\begin{tablenotes}[flushleft]",
  "\\footnotesize",
  "\\item \\textit{Notes:} The dependent variable is $\\ln(\\text{costs})$. The real exchange rate is the bilateral real exchange rate between the US and the supplier country, normalized to 1 for each country in 2015; an increase in the $RER$ variable indicates an appreciation of the foreign currency. Standard errors (clustered by make-model) are in parentheses below coefficients.",
  "Significance levels: \\sym{*} $p<0.10$, \\sym{**} $p<0.05$, \\sym{***} $p<0.01$.",
  "\\end{tablenotes}",
  "\\end{threeparttable}",
  "\\end{table}"
)

dir.create("cost_side/outputs", showWarnings = FALSE, recursive = TRUE)
writeLines(tex_lines, "cost_side/outputs/cost_reg_table.tex")
