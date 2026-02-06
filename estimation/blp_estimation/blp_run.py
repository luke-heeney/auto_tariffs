#########################################
# Script to run pyblp estimation on server
#########################################

# Import libraries

import numpy as np
import pandas as pd
import pyblp
import pickle
import datetime

from data_prep import build_product_data
from agent_prep import load_agent_data
from micromoments.division_moments2 import (
    DIVISION_NAME_TO_ID,
    build_division_micro_moments_pooled,
    build_division_micro_moments_yearly_range,
)

from micromoments.micro_income_price_mri import build_income_price_moments
from micromoments.micro_cex_moments import build_cex_moments
from micromoments.micro_second_choice_2015 import build_second_choice_moments_2015
from micromoments.micro_second_choice_2015_log import build_second_choice_moments_2015_log
from micromoments.micro_second_choice_2022_ev import build_second_choice_ev_moments_2022


# ==================================
# Choose Settings
# ==================================

# How to rescale observed product shares: None, 'halve', or 'double'
shares_scaling = 'double'  # options: None, 'halve', or 'double'

# Toggle using 45W subsidies data (0 = off, 1 = on)
include_45W_subsidies = 0

# Number of agents: here 400 per division (set in csv_path)
print(f"Num agents: 400 per div")

# Flag definitions (0 = off, 1 = on)
flag_values = {
    "include_us":                   0,
    "include_eu":                   1,
    "include_hyb_in_mpg":           1,  # Include hybrids in non-EV
    "include_ev_in_lux":            1,
    "include_ev_in_us":             1,
    "include_mpg":                  1,
    "include_size":                 0,
    "include_hp":                   1,
    "include_prob_moments":         1,
    "include_GH_instruments":       0,
    "include_old_models":           0,
    "include_price_mpg_int":        0,
    "include_outside_rc":           1,
    "first_log":                    0,
    "include_div_purchase_moments": 0,
    "include_div_ev_moments":       1,
    "include_div_type_moments":     1,
    "use_halton":                   1,
    "include_price":                0,
}

# Order matters for flags_bin (keep this list in sync if you add/remove flags)
flags_order = [
    "include_us",
    "include_eu",
    "include_hyb_in_mpg",
    "include_ev_in_lux",
    "include_ev_in_us",
    "include_mpg",
    "include_size",
    "include_hp",
    "include_prob_moments",
    "include_GH_instruments",
    "include_old_models",
    "include_price_mpg_int",
    "include_outside_rc",
    "include_div_purchase_moments",
    "include_div_ev_moments",
    "include_div_type_moments",
    "first_log",
    "use_halton",
    "include_price",
]

# Export individual flag variables (for existing code that expects them)
for name in flags_order:
    globals()[name] = flag_values[name]

# Tuple + binary string representation
flags = tuple(flag_values[name] for name in flags_order)
flags_bin = "".join("1" if f else "0" for f in flags)

# Nicely formatted logging
print("Flag configuration:")
for name in flags_order:
    print(f"  {name}: {flag_values[name]}")
print(f"  shares_scaling: {shares_scaling}")
print(f"  Flags binary: {flags_bin}")

# Timestamped results file name
now = datetime.datetime.now()
timestamp = now.strftime("%Y%m%d_%H%M%S")

subsidy_tag = "45W" if include_45W_subsidies else "subsidy"

if shares_scaling == "double":
    file_name = f"blp_results_{timestamp}_{flags_bin}_{subsidy_tag}_dbl.pkl"
else:
    file_name = f"blp_results_{timestamp}_{flags_bin}_{subsidy_tag}.pkl"

print(f"File Name: {file_name}")


# ========================================
# Import and preprocess product data, agent_data
# ========================================

# Build product data
product_data_path = (
    "data/blp_with_45W_subsidies_scale1p0.csv"
    if include_45W_subsidies == 1
    else "data/blp_with_subsidies.csv"
)

product_data = build_product_data(
    shares_scaling=shares_scaling,
    include_old_models=include_old_models,
    include_ev_in_us=include_ev_in_us,
    include_ev_in_lux=include_ev_in_lux,
    include_hyb_in_mpg=include_hyb_in_mpg,
    filepath=product_data_path,
)

# Add instrument
product_data["demand_instruments0"] = product_data["rer_lag1"]

# Define clusters
product_data.rename(columns={"product_ids": "clustering_ids"}, inplace=True)

product_data.to_csv(f"product_data_{subsidy_tag}.csv")

# Toggle regional divisions
use_division = int(
    include_div_purchase_moments
    or include_div_ev_moments
    or include_div_type_moments
)

# Build agent data (this now also creates div_1,... when use_division == 1)
agent_data, product_data, INCOME_MEAN, INCOME_STD = load_agent_data(
    product_data=product_data,
    first_log=first_log,
    division_map=DIVISION_NAME_TO_ID,
    use_division=use_division,
    csv_path="data/agent_incomes_400perdivision_updated.csv",
)

agent_data.to_csv(f"agent_data_{subsidy_tag}.csv")


# ========================================
# Create MICRO MOMENTS
# ========================================

# MRI 2019 income × price
micro_moments_income_price = build_income_price_moments(product_data)

# CEX: purchase probability & price differences (we ignore age moments)
(
    micro_moments_income_quintiles,
    micro_price_diff_pre,
    micro_price_diff_cov,
    micro_price_diff_pool,
    _,
    _,
) = build_cex_moments(agent_data, product_data)

# 2015 second-choice moments (log-mpg version)
second_choice_moments_2015 = build_second_choice_moments_2015_log(
    product_data,
    include_mpg=include_mpg,
    include_size=include_size,
    include_hp=include_hp,
    include_eu=include_eu,
    include_us=include_us,
    include_ev_in_lux=include_ev_in_lux,
    include_hyb_in_mpg=include_hyb_in_mpg,
)

# 2022 EV second-choice moments
ev_moments_2022 = build_second_choice_ev_moments_2022(product_data)

# State division moments: pooled + yearly (2021–2024)
division_purchase_moments_pooled = []
division_ev_moments_pooled = []
division_type_moments_pooled = []

division_purchase_moments_recent = []
division_ev_moments_recent = []
division_type_moments_recent = []
division_truck_moments_recent = []
division_suv_moments_recent = []

if include_div_purchase_moments or include_div_ev_moments or include_div_type_moments:
    # Year-specific division moments 2021–2024
    div_moms_recent = build_division_micro_moments_yearly_range(
        product_data=product_data,
        agent_data=agent_data,
        div_col_index=1,   # first division dummy in demographics: div_1
        base_dir="data",
        N_obs=1000,
        start_year=2021,
        end_year=2024,
    )

    if include_div_purchase_moments:
        division_purchase_moments_recent = div_moms_recent["purchase"]

    if include_div_ev_moments:
        division_ev_moments_recent = div_moms_recent["ev"]

    if include_div_type_moments:
        division_type_moments_recent = div_moms_recent["types"]
        division_truck_moments_recent = div_moms_recent["types_truck"]
        division_suv_moments_recent = div_moms_recent["types_suv"]


# Select micromoments
micro_moments = []

# Core CEX price moments
micro_moments += micro_price_diff_pool

# Second-choice structure moments
micro_moments += second_choice_moments_2015
micro_moments += ev_moments_2022

# Use YEAR-SPECIFIC division moments (2021–2024)
micro_moments += division_purchase_moments_recent
micro_moments += division_ev_moments_recent
micro_moments += division_truck_moments_recent
micro_moments += division_suv_moments_recent

# Income-based purchase probability moments (optional)
if include_prob_moments == 1:
    micro_moments += micro_moments_income_quintiles

# ========================================
# Run BLP on filtered moments
# ========================================

# === 1) Mean utility formulation ===

if include_hyb_in_mpg == 1:
    # VERSION: include hybrids in mpg effect
    X1_formulation = pyblp.Formulation(
        "1 + I(prices-subsidy) + log_size_std + log_hp_std + "
        "ln_mpg_icehyb + ln_mpg_ev + "
        "hybrid*C(market_ids) + ev*C(market_ids) + "
        "van_d + truck_d*C(market_ids) + suv_d*C(market_ids)",
        absorb="C(firm_ids)",
    )
else:
    # VERSION: mpg effect only for ICE vehicles
    X1_formulation = pyblp.Formulation(
        "1 + I(prices-subsidy) + log_size_std + log_hp_std + "
        "ln_mpg_ice + ln_mpg_hyb + "
        "hybrid*C(market_ids) + ev*C(market_ids) + "
        "van_d + truck_d + suv_d*C(market_ids)",
        absorb="C(firm_ids)",
    )

# === 2) Random-coefficient block ===
# Columns (X2 terms): [1, prices, ln_mpg_ice, log_size_std, log_hp_std,
#                      van_d, truck_d, suv_d, ev, euro_brand, us_brand, luxury_brand]
X2_formulation = pyblp.Formulation(
    "1 + I(prices-subsidy) + ln_mpg_ice + log_size_std + log_hp_std + "
    "van_d + truck_d + suv_d + ev + euro_brand + us_brand + luxury_brand"
)

# === 3) Agent formulation ===
if use_division:
    # Build division dummy terms I(div_1) + ... + I(div_6)
    unique_div_ids = sorted(set(DIVISION_NAME_TO_ID.values()))
    div_terms = " + ".join(f"I(div_{d})" for d in unique_div_ids)
    agent_formulation_str = f"0 + I(log_income_10k) + {div_terms}"
else:
    agent_formulation_str = "0 + I(log_income_10k)"

agent_formulation = pyblp.Formulation(agent_formulation_str)

product_formulations = (X1_formulation, X2_formulation)


# === 4) Define initial sigma and its bounds ===

# Zero sigma entries based on flags
sig_mpg   = 3.0 if include_mpg   == 1 else 0.0
sig_size  = 3.0 if include_size  == 1 else 0.0
sig_hp    = 3.0 if include_hp    == 1 else 0.0
sig_euro  = 2.0 if include_eu    == 1 else 0.0
sig_us    = 2.0 if include_us    == 1 else 0.0
sig_price = 1.0 if include_price == 1 else 0.0

# Allow turning on/off random coefficient on the outside option
outside_rc = 1.0 if include_outside_rc == 1 else 0.0

# Initial Σ (12x12)
initial_sigma = np.diag(
    [
        outside_rc,  # 1  (outside option random coefficient)
        sig_price,   # prices
        sig_mpg,     # ln_mpg_ice
        sig_size,    # log_size_std
        sig_hp,      # log_hp_std
        5.538,       # van_d
        6.309,       # truck_d
        3.617,       # suv_d
        3.65779,     # ev
        sig_euro,    # euro_brand
        sig_us,      # us_brand
        2.0,         # luxury_brand
    ]
)

S = initial_sigma.shape[0]
lb = -np.inf * np.ones_like(initial_sigma)
ub = np.inf * np.ones_like(initial_sigma)
i = np.arange(S)
lb[i, i] = 0.0
ub[i, i] = 200.0



# === 5) Initial Π ===
# Columns (X2 terms): [1, prices, ln_mpg_ice, log_size_std, log_hp_std,
#                      van_d, truck_d, suv_d, ev, euro_brand, us_brand, luxury_brand]
pi_rows = []

# Row 1: log income (for prob moments / price interactions)
mpg_pi  = 0.5 if include_price_mpg_int == 1 else 0.0
prob_pi = 2.2 if include_prob_moments == 1 else 0.0

pi_rows.append(
    [prob_pi, 3.629311, mpg_pi, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
)

if use_division:
    unique_div_ids = sorted(set(DIVISION_NAME_TO_ID.values()))

    # Explicit non-zero sets for each attribute
    EV_NONZERO_IDS = {
        DIVISION_NAME_TO_ID["North East"],
        DIVISION_NAME_TO_ID["South Atlantic"],
        DIVISION_NAME_TO_ID["Mountain"],
        DIVISION_NAME_TO_ID["Pacific"],
    }

    SUV_NONZERO_IDS = {
        DIVISION_NAME_TO_ID["North East"],
        DIVISION_NAME_TO_ID["North Central"],
    }

    TRUCK_NONZERO_IDS = {
        DIVISION_NAME_TO_ID["North Central"],
        DIVISION_NAME_TO_ID["South Central"],
        DIVISION_NAME_TO_ID["Mountain"],
    }

    for _div_id in unique_div_ids:
        row = [0.0] * 12

        # Intercept × division: North East as base (no intercept shift)
        if include_div_purchase_moments == 1 and _div_id != DIVISION_NAME_TO_ID["North East"]:
            row[0] = 0.5  # interaction on constant term

        # Truck & SUV division interactions
        if include_div_type_moments == 1:
            # truck_d (col 6): only for TRUCK_NONZERO_IDS
            if _div_id in TRUCK_NONZERO_IDS:
                row[6] = 1.0

            # suv_d (col 7): only for SUV_NONZERO_IDS
            if _div_id in SUV_NONZERO_IDS:
                row[7] = 1.0

        # EV division interactions
        if include_div_ev_moments == 1:
            # ev (col 8): only for EV_NONZERO_IDS
            if _div_id in EV_NONZERO_IDS:
                row[8] = 1.0

        pi_rows.append(row)

# Π must be (num_demographics × 12)
initial_pi = np.array(pi_rows).T




print("Initial Π (pi) matrix:")
print(initial_pi)
print("Π shape:", initial_pi.shape)


# === 6) Problem and solve ===

pyblp.options.verbose = True

problem = pyblp.Problem(
    product_formulations=product_formulations,
    product_data=product_data,
    agent_data=agent_data,
    agent_formulation=agent_formulation,
)

# with pyblp.parallel(24, use_pathos=True):
#     results = problem.solve(
#         sigma=initial_sigma,
#         pi=initial_pi,
#         micro_moments=micro_moments,
#         se_type="clustered",  # VERSION
#         W_type="clustered",
#         iteration=pyblp.Iteration("squarem", {"atol": 1e-13, "rtol": 1e-13}),
#         optimization=pyblp.Optimization("l-bfgs-b", {"gtol": 1e-6}),
#         sigma_bounds=(lb, ub),
#         shares_bounds=(1e-300, 1 - 1e-12),
#     )

# # Save results to pkl
# with open(f"pickles2/{file_name}", "wb") as f:  # VERSION: change name if desired
#     pickle.dump(results, f)
