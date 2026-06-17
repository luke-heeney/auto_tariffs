library(dplyr)
library(fixest)

normalize_country_vector <- function(x) {
  raw <- as.character(x)
  key <- toupper(raw)
  key <- gsub("[\u00a0]", " ", key, fixed = FALSE)
  key <- gsub("[._]", " ", key)
  key <- gsub("[^A-Z0-9/& -]+", " ", key)
  key <- gsub("\\s+", " ", trimws(key))

  aliases <- c(
    "UNITED STATES" = "United States",
    "UNITED STATES OF AMERICA" = "United States",
    "USA" = "United States",
    "US" = "United States",
    "U S" = "United States",
    "U S A" = "United States",
    "UNITED KINGDOM" = "United Kingdom",
    "GREAT BRITAIN" = "United Kingdom",
    "UK" = "United Kingdom",
    "GB" = "United Kingdom",
    "GBR" = "United Kingdom",
    "UNITED" = "United Kingdom",
    "MEXICO" = "Mexico",
    "M" = "Mexico",
    "MX" = "Mexico",
    "JAPAN" = "Japan",
    "J" = "Japan",
    "JP" = "Japan",
    "KOREA" = "Korea",
    "SOUTH KOREA" = "Korea",
    "K" = "Korea",
    "KR" = "Korea",
    "GERMANY" = "Germany",
    "G" = "Germany",
    "DE" = "Germany",
    "DEU" = "Germany",
    "CHINA" = "China",
    "CH" = "China",
    "CHN" = "China",
    "CANADA" = "Canada",
    "CN" = "Canada",
    "CA" = "Canada",
    "AUSTRIA" = "Austria",
    "A" = "Austria",
    "AU" = "Austria",
    "AT" = "Austria",
    "BELGIUM" = "Belgium",
    "BE" = "Belgium",
    "BRAZIL" = "Brazil",
    "BR" = "Brazil",
    "CZECHIA" = "Czechia",
    "CZECH REPUBLIC" = "Czechia",
    "CZ" = "Czechia",
    "DENMARK" = "Denmark",
    "DK" = "Denmark",
    "FINLAND" = "Finland",
    "FN" = "Finland",
    "FRANCE" = "France",
    "F" = "France",
    "FR" = "France",
    "HUNGARY" = "Hungary",
    "H" = "Hungary",
    "HUN" = "Hungary",
    "INDIA" = "India",
    "IN" = "India",
    "IND" = "India",
    "ITALY" = "Italy",
    "I" = "Italy",
    "NETHERLANDS" = "Netherlands",
    "N" = "Netherlands",
    "PHILIPPINES" = "Philippines",
    "P" = "Philippines",
    "POLAND" = "Poland",
    "PL" = "Poland",
    "PORTUGAL" = "Portugal",
    "PO" = "Portugal",
    "SLOVAKIA" = "Slovakia",
    "SL" = "Slovakia",
    "SPAIN" = "Spain",
    "SP" = "Spain",
    "ESP" = "Spain",
    "SWEDEN" = "Sweden",
    "SW" = "Sweden",
    "SE" = "Sweden",
    "THAILAND" = "Thailand",
    "TH" = "Thailand",
    "TURKEY" = "Turkey",
    "T" = "Turkey",
    "VIETNAM" = "Vietnam",
    "SERBIA" = "Serbia",
    "SRB" = "Serbia",
    "SOUTH AFRICA" = "South Africa",
    "REPUBLIC OF SOUTH AFRICA" = "South Africa",
    "SA" = "South Africa",
    "SAF" = "South Africa",
    "AF" = "South Africa",
    "AUSTRALIA" = "Australia",
    "AUS" = "Australia",
    "TAIWAN" = "Taiwan",
    "TW" = "Taiwan",
    "OTHER" = "Other",
    "OT" = "Other"
  )

  out <- raw
  hit <- !is.na(key) & key %in% names(aliases)
  out[hit] <- unname(aliases[key[hit]])
  out[is.na(raw) | key %in% c("", "<NA>", "NA", "NAN", "NONE", "NULL")] <- NA_character_
  out
}

resolve_cost_side_path <- function(file_name) {
  if (file.exists(file_name)) {
    file_name
  } else {
    file.path("cost_side", file_name)
  }
}

ensure_outputs_dir <- function() {
  out_dir <- "cost_side/outputs"
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  out_dir
}

load_cost_panel <- function(file_name) {
  df <- read.csv(resolve_cost_side_path(file_name), stringsAsFactors = FALSE) %>%
    mutate(
      product_ids = as.character(product_ids),
      market_year = as.integer(market_year),
      year = as.integer(year),
      make_model = as.character(make_model),
      plant_country = normalize_country_vector(plant_country),
      pcOth1_code1 = normalize_country_vector(pcOth1_code1),
      across(
        c(costs, markups, rer_pcOth1_code1_n2015, pcOth1_pct1, pcOth1_pct2, size, weight, hp, mpg),
        ~ suppressWarnings(as.numeric(.))
      )
    )
  logical_cols <- intersect(
    c(
      "uses_collapsed_duplicate_row",
      "is_us_assembled",
      "is_foreign_assembled",
      "source_country_stable",
      "has_direct_primary_country",
      "primary_source_country_consistent",
      "canonical_exposure_ok"
    ),
    names(df)
  )
  for (col in logical_cols) {
    df[[col]] <- as.logical(df[[col]])
  }
  df
}

load_blp_prices <- function() {
  price_path <- resolve_cost_side_path("../post_est/data/raw/blpUS0804.csv")
  read.csv(price_path, stringsAsFactors = FALSE) %>%
    transmute(
      product_ids = as.character(product_ids),
      market_year = as.integer(market_year),
      prices = suppressWarnings(as.numeric(prices))
    ) %>%
    filter(!is.na(product_ids), !is.na(market_year), !is.na(prices), prices > 0) %>%
    distinct(product_ids, market_year, .keep_all = TRUE)
}

load_elasticities <- function() {
  elas_path <- resolve_cost_side_path("../post_est/data/derived/product_year_elasticities.csv")
  read.csv(elas_path, stringsAsFactors = FALSE) %>%
    transmute(
      product_ids = as.character(product_ids),
      market_year = as.integer(market_year),
      own_elas_t = suppressWarnings(as.numeric(own_elasticity))
    )
}

load_normalized_exchange_lookup <- function() {
  exchange_path <- resolve_cost_side_path("../processed_data/exchange_rates/exchange_rates.csv")

  exchange <- read.csv(
    exchange_path,
    stringsAsFactors = FALSE,
    check.names = FALSE,
    fileEncoding = "UTF-8-BOM"
  )
  exchange_country_col <- names(exchange)[1]
  names(exchange)[names(exchange) == exchange_country_col] <- "country"
  exchange$country <- normalize_country_vector(exchange$country)
  exchange <- exchange %>%
    filter(!is.na(country)) %>%
    distinct(country, .keep_all = TRUE)

  drop_cols <- intersect(c("Country Code", "2010", "2011", "2012", "2013"), names(exchange))
  exchange <- exchange %>% select(-all_of(drop_cols))

  year_cols <- setdiff(names(exchange), "country")
  exchange[year_cols] <- lapply(exchange[year_cols], function(col) suppressWarnings(as.numeric(col)))

  if (!("2015" %in% year_cols)) {
    stop("Exchange-rate file is missing the 2015 normalization column.")
  }

  exchange[year_cols] <- lapply(exchange[year_cols], function(col) col / exchange[["2015"]])

  lookup_rows <- lapply(year_cols, function(year_col) {
    data.frame(
      country = exchange$country,
      year = as.integer(year_col),
      ln_inv_rer = -log(exchange[[year_col]]),
      stringsAsFactors = FALSE
    )
  })

  bind_rows(lookup_rows) %>%
    filter(!is.na(country), !is.na(year), is.finite(ln_inv_rer))
}

attach_direct_future_rer <- function(
  df,
  exchange_lookup,
  code_col = "pcOth1_code1",
  year_col = "year",
  out_col = "ln_inv_rer_code1_tplus1"
) {
  future_keys <- data.frame(
    .row_id = seq_len(nrow(df)),
    country = as.character(df[[code_col]]),
    year = as.integer(df[[year_col]]) + 1L,
    stringsAsFactors = FALSE
  )

  matched <- future_keys %>%
    left_join(exchange_lookup, by = c("country", "year")) %>%
    arrange(.row_id)

  out <- df
  out[[out_col]] <- matched$ln_inv_rer
  out
}

build_regression_frame <- function(panel_df, include_elasticities = FALSE, include_prices = FALSE) {
  df <- panel_df %>%
    filter(costs > 0, rer_pcOth1_code1_n2015 > 0) %>%
    arrange(make_model, year) %>%
    group_by(make_model) %>%
    mutate(
      observed_foreign_share = coalesce(pcOth1_pct1, 0) + coalesce(pcOth1_pct2, 0),
      pcOth1_pct1_lag1 = lag(pcOth1_pct1, 1),
      observed_foreign_share_lag1 = lag(observed_foreign_share, 1)
    ) %>%
    ungroup() %>%
    filter(
      !is.na(size), !is.na(weight), !is.na(hp), !is.na(mpg)
    ) %>%
    mutate(
      ln_costs = log(costs),
      ln_markups = log(markups),
      ln_inv_rer_code1 = -log(rer_pcOth1_code1_n2015),
      ln_size = log(size),
      ln_weight = log(weight),
      ln_hp = log(hp),
      ln_mpg = log(mpg)
    ) %>%
    ungroup()

  if (include_prices) {
    prices <- load_blp_prices()
    df <- df %>%
      left_join(prices, by = c("product_ids", "market_year")) %>%
      mutate(ln_prices = log(prices))
  }

  if (!include_elasticities) {
    return(df)
  }

  elas <- load_elasticities()
  df %>%
    left_join(elas, by = c("product_ids", "market_year")) %>%
    arrange(make_model, year) %>%
    group_by(make_model) %>%
    mutate(
      own_elas_lag1 = lag(own_elas_t, 1)
    ) %>%
    ungroup() %>%
    mutate(
      log_abs_own_elas_t = log(abs(own_elas_t)),
      log_abs_own_elas_lag1 = log(abs(own_elas_lag1))
    ) %>%
    filter(
      !is.na(own_elas_t),
      !is.na(own_elas_lag1),
      is.finite(log_abs_own_elas_t),
      is.finite(log_abs_own_elas_lag1)
    )
}

build_fd_frame <- function(df, keep_cols = character()) {
  df %>%
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
    transmute(
      make_model = make_model,
      year = year,
      ln_costs = d_ln_costs,
      ln_inv_rer_code1 = d_ln_inv_rer_code1,
      !!!rlang::syms(keep_cols),
      ln_size = d_ln_size,
      ln_weight = d_ln_weight,
      ln_hp = d_ln_hp,
      ln_mpg = d_ln_mpg
    )
}

sample_summary <- function(df, sample_name) {
  data.frame(
    sample = sample_name,
    rows = nrow(df),
    make_models = dplyr::n_distinct(df$make_model),
    years = dplyr::n_distinct(df$year),
    source_countries = dplyr::n_distinct(df$pcOth1_code1),
    stringsAsFactors = FALSE
  )
}

extract_term_row <- function(model, term, sample_name, spec_name) {
  ct <- summary(model)$coeftable
  coef_names <- rownames(ct)
  if (is.null(coef_names)) {
    coef_names <- names(coef(model))
  }

  matched_term <- term
  if (!(matched_term %in% coef_names) && grepl(":", term, fixed = TRUE)) {
    requested_parts <- sort(strsplit(term, ":", fixed = TRUE)[[1]])
    candidate_terms <- coef_names[
      vapply(
        strsplit(coef_names, ":", fixed = TRUE),
        function(parts) identical(sort(parts), requested_parts),
        logical(1)
      )
    ]
    if (length(candidate_terms) == 1) {
      matched_term <- candidate_terms[[1]]
    }
  }

  if (!(matched_term %in% coef_names)) {
    stop(paste("Missing coefficient in model:", term))
  }

  term_idx <- which(coef_names == matched_term)[1]
  data.frame(
    sample = sample_name,
    spec = spec_name,
    coefficient = matched_term,
    estimate = unname(ct[term_idx, "Estimate"]),
    std_error = unname(ct[term_idx, "Std. Error"]),
    nobs = nobs(model),
    stringsAsFactors = FALSE
  )
}

relabel_model_numbers <- function(path, labels) {
  lines <- readLines(path, warn = FALSE)
  model_idx <- grep("^\\s*Model:\\s*&", lines)
  if (length(model_idx) == 1) {
    label_str <- paste(labels, collapse = "            & ")
    lines[model_idx] <- sub(
      "&.*\\\\\\\\\\s*$",
      paste0("& ", label_str, "\\\\\\\\  "),
      lines[model_idx]
    )
  }
  writeLines(lines, path)
}

append_table_note <- function(path, note) {
  lines <- readLines(path, warn = FALSE)
  end_idx <- grep("^\\\\end\\{table\\}", lines)
  if (length(end_idx) != 1) {
    stop(paste("Could not find a unique end{table} in", path))
  }
  note_lines <- c(
    "\\vspace{0.25em}",
    "\\begin{minipage}{0.96\\textwidth}",
    paste0("\\footnotesize \\textit{Notes:} ", note),
    "\\end{minipage}"
  )
  lines <- append(lines, note_lines, after = end_idx - 1L)
  writeLines(lines, path)
}
