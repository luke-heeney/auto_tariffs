library(dplyr)
library(fixest)

panel_path <- if (file.exists("cost_side_panel_dropped.csv")) {
  "cost_side_panel_dropped.csv"
} else {
  "cost_side/cost_side_panel_dropped.csv"
}

out_dir <- "cost_side/outputs"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

final_df <- read.csv(panel_path)

map_vehicle_type <- function(vehicle_type, make_model) {
  vt <- toupper(trimws(as.character(vehicle_type)))
  mm <- tolower(trimws(as.character(make_model)))

  is_van_name <- grepl(
    "van|minivan|transit|sprinter|promaster|pro_master|pacifica|caravan|voyager|odyssey|sienna",
    mm
  )

  if (is.na(vt) || vt == "" || vt == "NA" || vt == "NAN") {
    return(NA_character_)
  }

  if (vt %in% c("PC", "CAR", "SEDAN", "COUPE", "HATCHBACK")) {
    return("car")
  }

  if (vt %in% c("TRUCK", "PICKUP", "LT")) {
    if (is_van_name) return("van")
    return("truck")
  }

  if (vt %in% c("SUV", "CUV", "UTILITY", "MPV")) {
    if (is_van_name) return("van")
    return("suv")
  }

  if (vt %in% c("VAN", "MINIVAN")) {
    return("van")
  }

  if (is_van_name) {
    return("van")
  }

  NA_character_
}

# ---- Build df ----
df_raw <- final_df %>%
  transmute(
    costs,
    rer_pcOth1_code1_n2015,
    pcOth1_pct1,
    vehicle_type,
    make_model = as.character(make_model),
    year = as.integer(year),
    size, weight, hp, mpg
  ) %>%
  mutate(across(
    c(costs, rer_pcOth1_code1_n2015, pcOth1_pct1, size, weight, hp, mpg),
    ~ suppressWarnings(as.numeric(.))
  )) %>%
  mutate(
    veh_type4 = vapply(
      seq_len(n()),
      function(i) map_vehicle_type(vehicle_type[i], make_model[i]),
      FUN.VALUE = character(1)
    )
  )

df <- df_raw %>%
  arrange(make_model, year) %>%
  group_by(make_model) %>%
  mutate(pcOth1_pct1_lag1 = lag(pcOth1_pct1, 1)) %>%
  ungroup() %>%
  filter(costs > 0, rer_pcOth1_code1_n2015 > 0) %>%
  filter(
    !is.na(pcOth1_pct1_lag1),
    !is.na(size), !is.na(weight), !is.na(hp), !is.na(mpg),
    !is.na(veh_type4)
  ) %>%
  mutate(
    veh_type4 = factor(veh_type4, levels = c("car", "truck", "suv", "van")),
    ln_costs = log(costs),
    ln_inv_rer_code1 = -log(rer_pcOth1_code1_n2015),
    ln_size = log(size),
    ln_weight = log(weight),
    ln_hp = log(hp),
    ln_mpg = log(mpg),
    rho_rer = ln_inv_rer_code1 * pcOth1_pct1_lag1,
    rho_rer_car = rho_rer * as.numeric(veh_type4 == "car"),
    rho_rer_truck = rho_rer * as.numeric(veh_type4 == "truck"),
    rho_rer_suv = rho_rer * as.numeric(veh_type4 == "suv"),
    rho_rer_van = rho_rer * as.numeric(veh_type4 == "van")
  )

# ---- Levels regressions ----
m1_levels <- feols(
  ln_costs ~ rho_rer_car + rho_rer_truck + rho_rer_suv + rho_rer_van +
    ln_size + ln_weight + ln_hp + ln_mpg | make_model + year,
  data = df,
  cluster = ~ make_model
)

m2_levels <- feols(
  ln_costs ~ rho_rer_car + rho_rer_truck + rho_rer_suv + rho_rer_van | make_model + year,
  data = df,
  cluster = ~ make_model
)

# ---- First differences (consecutive years) ----
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
    d_ln_mpg = ln_mpg - lag(ln_mpg)
  ) %>%
  ungroup() %>%
  filter(year_gap == 1) %>%
  filter(!is.na(d_ln_costs), !is.na(d_ln_inv_rer_code1)) %>%
  mutate(
    rho_rer_fd = d_ln_inv_rer_code1 * pcOth1_pct1_lag1,
    rho_rer_fd_car = rho_rer_fd * as.numeric(veh_type4 == "car"),
    rho_rer_fd_truck = rho_rer_fd * as.numeric(veh_type4 == "truck"),
    rho_rer_fd_suv = rho_rer_fd * as.numeric(veh_type4 == "suv"),
    rho_rer_fd_van = rho_rer_fd * as.numeric(veh_type4 == "van")
  )

m1_fd <- feols(
  d_ln_costs ~ rho_rer_fd_car + rho_rer_fd_truck + rho_rer_fd_suv + rho_rer_fd_van +
    d_ln_size + d_ln_weight + d_ln_hp + d_ln_mpg | year,
  data = df_fd,
  cluster = ~ make_model
)

m2_fd <- feols(
  d_ln_costs ~ rho_rer_fd_car + rho_rer_fd_truck + rho_rer_fd_suv + rho_rer_fd_van | year,
  data = df_fd,
  cluster = ~ make_model
)

# ---- Diagnostics ----
levels_counts <- df %>%
  count(veh_type4, name = "n_levels_sample")

fd_counts <- df_fd %>%
  count(veh_type4, name = "n_fd_sample")

diag_df <- full_join(levels_counts, fd_counts, by = "veh_type4") %>%
  mutate(
    n_levels_sample = ifelse(is.na(n_levels_sample), 0L, n_levels_sample),
    n_fd_sample = ifelse(is.na(n_fd_sample), 0L, n_fd_sample)
  ) %>%
  arrange(match(veh_type4, c("car", "truck", "suv", "van")))

write.csv(diag_df, file.path(out_dir, "cost_reg_vehicle_type_sample_counts.csv"), row.names = FALSE)

# ---- Export LaTeX table ----
etable(
  m1_levels, m2_levels, m1_fd, m2_fd,
  tex = TRUE,
  file = file.path(out_dir, "cost_reg_vehicle_type_table.tex"),
  replace = TRUE,
  title = "Cost-side regressions with vehicle-type-specific exchange-rate pass-through",
  label = "tab:cost_reg_vehicle_type",
  dict = c(
    "rho_rer_car" = "$\\rho_{j,t-1}\\times\\log(RER_{jt}) \\times \\mathbb{1}[car]$",
    "rho_rer_truck" = "$\\rho_{j,t-1}\\times\\log(RER_{jt}) \\times \\mathbb{1}[truck]$",
    "rho_rer_suv" = "$\\rho_{j,t-1}\\times\\log(RER_{jt}) \\times \\mathbb{1}[suv]$",
    "rho_rer_van" = "$\\rho_{j,t-1}\\times\\log(RER_{jt}) \\times \\mathbb{1}[van]$",
    "rho_rer_fd_car" = "$\\Delta\\log(RER_{jt})\\times\\rho_{j,t-1} \\times \\mathbb{1}[car]$",
    "rho_rer_fd_truck" = "$\\Delta\\log(RER_{jt})\\times\\rho_{j,t-1} \\times \\mathbb{1}[truck]$",
    "rho_rer_fd_suv" = "$\\Delta\\log(RER_{jt})\\times\\rho_{j,t-1} \\times \\mathbb{1}[suv]$",
    "rho_rer_fd_van" = "$\\Delta\\log(RER_{jt})\\times\\rho_{j,t-1} \\times \\mathbb{1}[van]$",
    "ln_size" = "$\\ln(\\text{size})$",
    "ln_weight" = "$\\ln(\\text{weight})$",
    "ln_hp" = "$\\ln(\\text{hp})$",
    "ln_mpg" = "$\\ln(\\text{mpg})$",
    "d_ln_size" = "$\\Delta\\ln(\\text{size})$",
    "d_ln_weight" = "$\\Delta\\ln(\\text{weight})$",
    "d_ln_hp" = "$\\Delta\\ln(\\text{hp})$",
    "d_ln_mpg" = "$\\Delta\\ln(\\text{mpg})$"
  ),
  fitstat = ~n + r2 + wr2,
  signif.code = c("***" = 0.01, "**" = 0.05, "*" = 0.10)
)

cat("Saved:\n")
cat(" -", file.path(out_dir, "cost_reg_vehicle_type_table.tex"), "\n")
cat(" -", file.path(out_dir, "cost_reg_vehicle_type_sample_counts.csv"), "\n")
