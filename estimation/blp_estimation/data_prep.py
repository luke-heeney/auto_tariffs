# data_prep.py

import numpy as np
import pandas as pd


# ========================================
# Core loader: read CSV + basic cleaning
# ========================================

def load_product_data(
    filepath: str = "data/blp_with_subsidies.csv",
    shares_scaling: str | None = None,  # None, "halve", "double"
    include_old_models: int = 0
) -> pd.DataFrame:
    """
    Load and pre-clean product data.

    - Applies core filters on shares, prices, mpg, weight.
    - Scales prices, size, mpg, hp, weight.
    - Constructs hpwt, ev, hybrid.
    - Optionally rescales observed product shares.
    - Optionally drops non-current-year models.
    """
    df = pd.read_csv(filepath)

    # Basic filters
    df = df[df["shares"] > 0.00001]
    df = df[df["prices"] < 100000]  # prices in USD; filter at 100k
    if shares_scaling == "double":
        df = df[df["prices"] < 100000]  # prices in USD; filter at 100k
        df = df[df["shares"] > 0.00002]
    df = df[~df["mpg"].isna()]
    df = df[~df["weight"].isna()]

    # Scale variables
    df["prices"] = df["prices"] / 100000  # prices in $100k units
    df["subsidy"] = df["subsidy"].fillna(0.0)
    df["subsidy"] = df["subsidy"] / 100000  # prices in $100k units
    df["size"] = df["size"] * 0.006944444 / 100  # sq in → sq ft / 100
    df["mpg"] = df["mpg"] / 10  # mpg in 10s
    df["hp"] = df["hp"] / 100  # hp in 100s
    df["weight"] = df["weight"] / 1000  # 1000s of pounds
    df["hpwt"] = df["hp"] / df["weight"]

    # EV / hybrid dummies
    df["ev"] = (df["engine_type"] == "Electric").astype(int)
    df["hybrid"] = (df["engine_type"] == "Hybrid").astype(int)

    # Drop to current-year models only if requested
    if include_old_models == 0:
        df = df[df["market_ids"] == df["product_year"]]

    # Optional share scaling (micro robustness: halve / double shares)
    if shares_scaling is not None:
        if shares_scaling == "halve":
            df["shares"] *= 0.5
        elif shares_scaling == "double":
            df["shares"] *= 2.4
        else:
            raise ValueError("shares_scaling must be None, 'halve', or 'double'.")

    return df


# ========================================
# Brand-region: EU vs US
# ========================================

def add_brand_region_dummies(
    product_data: pd.DataFrame,
    include_ev_in_us: int = 1,
) -> pd.DataFrame:
    """
    Add euro_brand / us_brand dummy columns based on firm_ids.
    Also creates an intermediate firm_region assignment.

    include_ev_in_us:
        - 1: EV-only brands (Tesla/Rivian/Lucid) counted as US brands
        - 0: EV brands treated as OTHER
    """
    df = product_data.copy()

    EU_BRANDS = {
        "audi", "bmw", "mercedesbenz", "mini", "volkswagen", "porsche",
        "volvo", "polestar", "jaguar", "landrover", "fiat", "alfaromeo",
        "maserati", "lotus", "smart",
    }
    if include_ev_in_us == 1:
        US_BRANDS = {
            "buick", "cadillac", "chevrolet", "chrysler", "dodge", "ford",
            "gmc", "jeep", "lincoln", "ram", "tesla", "rivian", "lucidmotors",
        }
    else:
        US_BRANDS = {
            "buick", "cadillac", "chevrolet", "chrysler", "dodge", "ford",
            "gmc", "jeep", "lincoln", "ram",
        }

    def brand_region(b: str) -> str:
        if b in EU_BRANDS:
            return "EU"
        if b in US_BRANDS:
            return "US"
        return "OTHER"

    # Map firm_ids → brand_region via brand name (assuming firm_ids is a brand-ish string)
    firm_str = (
        df["firm_ids"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.replace(" ", "", regex=False)
    )
    df["brand_region"] = firm_str.map(brand_region)

    # Optional place for manual overrides at the firm_ids level
    FIRM_REGION_OVERRIDE: dict[str, str] = {
        # e.g. "toyota": "OTHER"
    }

    def majority_region(g: pd.Series) -> str:
        counts = g.value_counts()
        for pref in ["EU", "US"]:
            if pref in counts.index:
                return pref
        return "OTHER"

    firm_region_inferred = (
        df.groupby("firm_ids")["brand_region"]
        .apply(majority_region)
        .to_dict()
    )
    firm_region = {**firm_region_inferred, **FIRM_REGION_OVERRIDE}

    df["firm_region"] = df["firm_ids"].map(firm_region).fillna("OTHER")
    df["euro_brand"] = (df["firm_region"] == "EU").astype(int)
    df["us_brand"] = (df["firm_region"] == "US").astype(int)

    return df


# ========================================
# Vehicle-type, MPG, z-scores, luxury
# ========================================

def _zscore(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    m = s.mean(skipna=True)
    sd = s.std(skipna=True)
    if sd == 0 or np.isnan(sd):
        return s * 0.0
    return (s - m) / sd


def add_vehicle_and_mpg_features(
    product_data: pd.DataFrame,
    include_hyb_in_mpg: int = 0,
    include_ev_in_lux: int = 1,
) -> pd.DataFrame:
    """
    Add:su
      - suv_d, van_d, truck_d
      - ln_mpg_icehyb, ln_mpg_ice, ln_mpg_ev, ln_mpg_hyb
      - standardized log(size/hp/hpwt/mpg) variables
      - luxury_brand dummy (firm_ids-based, with optional EV inclusion)
    """
    df = product_data.copy()

    # --- vehicle-type dummies (car is base) ---
    vt = df["vehicle_type"].fillna("").str.lower().str.strip()
    df["suv_d"] = (vt == "suv").astype(float)
    df["van_d"] = (vt == "van").astype(float)
    df["truck_d"] = (vt == "truck").astype(float)

    # --- ln(mpg) decompositions ---
    df["mpg"] = pd.to_numeric(df["mpg"], errors="coerce")
    is_ev = df["ev"] == 1
    is_hyb = df["hybrid"] == 1
    is_evhyb = is_ev | is_hyb

    ln_mpg_all = np.log(df["mpg"].clip(lower=1e-6))

    # ICE + hybrid
    df["ln_mpg_icehyb"] = np.where(~is_ev, ln_mpg_all, 0.0)
    # ICE only
    df["ln_mpg_ice"] = np.where(~is_evhyb, ln_mpg_all, 0.0)
    # EV only
    df["ln_mpg_ev"] = np.where(is_ev, ln_mpg_all, 0.0)
    # Hybrids only
    df["ln_mpg_hyb"] = np.where(is_hyb, ln_mpg_all, 0.0)

    # --- standardized logs for continuous chars ---
    df["log_size_std"] = _zscore(np.log(df["size"].clip(lower=1e-6)))
    df["log_hp_std"] = _zscore(np.log(df["hp"].clip(lower=1e-6)))
    df["log_hpwt_std"] = _zscore(np.log(df["hpwt"].clip(lower=1e-6)))
    df["log_weight_std"] = _zscore(np.log(df["weight"].clip(lower=1e-6)))

    # mpg random-coefficient feature
    if include_hyb_in_mpg == 1:
        df["ln_mpg_std"] = _zscore(df["ln_mpg_icehyb"])
    else:
        df["ln_mpg_std"] = _zscore(df["ln_mpg_ice"])

    # EV / hybrid mpg features (mean-utility only)
    df["ln_mpg_ev_std"] = _zscore(df["ln_mpg_ev"])
    df["ln_mpg_hyb_std"] = _zscore(df["ln_mpg_hyb"])

    # --- luxury brand dummy from firm_ids ---
    firm_clean = (
        df["firm_ids"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.strip()
    )

    if include_ev_in_lux == 1:
        luxury_brands = {
            "porsche", "maserati", "lotus", "audi", "bmw", "mercedesbenz",
            "lexus", "infiniti", "cadillac", "lincoln", "jaguar", "landrover",
            "tesla", "rivian", "lucidmotors", "polestar",
            "acura", "volvo", "alpharomeo", "genesis",
        }
    else:
        luxury_brands = {
            "porsche", "maserati", "lotus", "audi", "bmw", "mercedesbenz",
            "lexus", "infiniti", "cadillac", "lincoln", "jaguar", "landrover",
            "acura", "volvo", "genesis", "alpharomeo",
        }

    df["luxury_brand"] = firm_clean.isin(luxury_brands).astype(float)

    return df


# ========================================
# High-level wrapper: one call from main script
# ========================================

def build_product_data(
    shares_scaling: str | None,
    include_old_models: int,
    include_ev_in_us: int,
    include_ev_in_lux: int,
    include_hyb_in_mpg: int,
    filepath: str = "data/blp_with_subsidies.csv",
) -> pd.DataFrame:
    """
    Full product-data pipeline:

      1. load_product_data(...)
      2. add_brand_region_dummies(...)
      3. add_vehicle_and_mpg_features(...)

    Returns a fully-prepared product_data DataFrame ready for BLP.
    """
    df = load_product_data(
        filepath=filepath,
        shares_scaling=shares_scaling,
        include_old_models=include_old_models,
    )
    df = add_brand_region_dummies(
        product_data=df,
        include_ev_in_us=include_ev_in_us,
    )
    df = add_vehicle_and_mpg_features(
        product_data=df,
        include_hyb_in_mpg=include_hyb_in_mpg,
        include_ev_in_lux=include_ev_in_lux,
    )
    return df
    

