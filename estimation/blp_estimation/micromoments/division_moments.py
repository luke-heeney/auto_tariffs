# division_moments.py
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import pyblp

# -------------------------------------------------------------------
# 1. Division IDs (used in agent_data['division_id'] and moments)
# -------------------------------------------------------------------

DIVISION_NAME_TO_ID: Dict[str, int] = {
    "North East":                         1,
    "North Central":                      2,
    "South Atlantic":                     3,
    "South Central":                      4,
    "Mountain":                           5,
    "Pacific":                            6,
}

# Mapping from original 9 Census division names (in CSV files) to condensed 6 names
CENSUS_DIV_TO_CONDENSED: Dict[str, str] = {
    "New England":        "North East",
    "Middle Atlantic":    "North East",
    "East North Central": "North Central",
    "West North Central": "North Central",
    "South Atlantic":     "South Atlantic",
    "East South Central": "South Central",
    "West South Central": "South Central",
    "Mountain":           "Mountain",
    "Pacific":            "Pacific",
}

# -------------------------------------------------------------------
# 2. Small helpers
# -------------------------------------------------------------------

def _ratio(v: np.ndarray) -> float:
    num, den = v[0], v[1] + 1e-16
    return num / den


def _ratio_grad(v: np.ndarray) -> np.ndarray:
    num, den = v[0], v[1] + 1e-16
    return np.array([1.0 / den, -num / (den ** 2)], dtype=float)


def _division_indicator(a, division_id, first_div_col, n_divisions=9):
    """
    Agent-side indicator 1{division == division_id} when demographics contain
    division dummies [div_1, ..., div_n] starting at column first_div_col.

    Results for each (first_div_col, n_divisions) pair are cached on the agent
    input so that multiple MicroParts (numerator/denominator) reuse the same
    slice instead of copying the block repeatedly.
    """
    cache_key = (first_div_col, n_divisions)
    cache_attr = "_division_indicator_cache"
    cache = getattr(a, cache_attr, None)

    if cache is None or cache.get("key") != cache_key:
        div_block = a.demographics[:, first_div_col : first_div_col + n_divisions]
        cache = {"key": cache_key, "block": div_block, "vectors": {}}
        setattr(a, cache_attr, cache)

    vectors = cache["vectors"]
    if division_id not in vectors:
        vectors[division_id] = cache["block"][:, division_id - 1]

    return vectors[division_id]


def _product_vec_inside(p) -> np.ndarray:
    """[0, 1, 1, ..., 1] – inside options only (j > 0)."""
    return np.r_[0.0, np.ones(p.size, dtype=float)]


def _product_vec_all(p) -> np.ndarray:
    """[1, 1, 1, ..., 1] – outside + inside."""
    return np.r_[1.0, np.ones(p.size, dtype=float)]


def _ev_vec_inside(p) -> np.ndarray:
    """[0, ev_1, ..., ev_J] – product EV dummy, outside = 0."""
    ev = np.asarray(p.ev, dtype=float).reshape(-1)
    return np.r_[0.0, ev]


def _vt_vec_inside(p, vt_name: str) -> np.ndarray:
    """
    Build vehicle-type indicator using dummy fields in products:
        - van_d
        - truck_d
        - suv_d
    and treat 'Car' as the residual category:
        car_d = 1 - (van_d + truck_d + suv_d).

    Returns [0, vt_1, ..., vt_J] so that outside option = 0.
    """
    vt = vt_name.strip().upper()

    # these must be present in product_data before creating the Problem
    van   = np.asarray(p.van_d,   dtype=float).reshape(-1)
    truck = np.asarray(p.truck_d, dtype=float).reshape(-1)
    suv   = np.asarray(p.suv_d,   dtype=float).reshape(-1)

    if vt == "VAN":
        vec = van
    elif vt == "TRUCK":
        vec = truck
    elif vt == "SUV":
        vec = suv
    elif vt == "CAR":
        # residual category: anything that's not van/truck/suv
        vec = 1.0 - np.clip(van + truck + suv, 0.0, 1.0)
    else:
        raise ValueError(f"Unknown vehicle_type '{vt_name}' in division vt moments.")

    # prepend 0 for the outside option
    return np.r_[0.0, vec]


# -------------------------------------------------------------------
# 3. Load division-level targets from CSVs
# -------------------------------------------------------------------

def _load_division_targets(base_dir: str):
    """Return three dataframes with year-division targets."""
    purchase = pd.read_csv(os.path.join(base_dir, "prob_purchase_given_division.csv"))
    ev       = pd.read_csv(os.path.join(base_dir, "prob_EV_given_division.csv"))
    vt       = pd.read_csv(os.path.join(base_dir, "prob_vehicle_type_given_division.csv"))

    # Standardize column names
    for df in (purchase, ev, vt):
        if "market_year" in df.columns:
            df.rename(columns={"market_year": "market_ids"}, inplace=True)

    # Map Census division names -> condensed names -> numeric id
    for df in (purchase, ev, vt):
        # First map old Census names to condensed names
        #df["division"] = df["division"].map(CENSUS_DIV_TO_CONDENSED)
        # Then map condensed names to numeric IDs
        df["division_id"] = df["division"].map(DIVISION_NAME_TO_ID)
        df.dropna(subset=["division_id"], inplace=True)

    purchase["division_id"] = purchase["division_id"].astype(int)
    ev["division_id"]       = ev["division_id"].astype(int)
    vt["division_id"]       = vt["division_id"].astype(int)

    return purchase, ev, vt


# -------------------------------------------------------------------
# 4. Build MicroMoments for divisions (year-specific)
# -------------------------------------------------------------------

def build_division_micro_moments(
    product_data: pd.DataFrame,
    agent_data: pd.DataFrame,
    div_col_index: int,
    base_dir: str = "moments_output",
    N_obs: int = 1000,
) -> Dict[str, List[pyblp.MicroMoment]]:
    """
    Build three lists of pyblp.MicroMoment objects:

        'purchase': P(purchase | division, year)
        'ev':       P(EV | purchase, division, year)
        'types':    P(vehicle_type | purchase, division, year)

    Parameters
    ----------
    product_data : DataFrame used in pyblp.Problem
        Must contain 'market_ids', 'ev', and 'vehicle_type'.

    agent_data : DataFrame used in pyblp.Problem
        Must contain demographic column (via agent_formulation) corresponding to
        division_id at position div_col_index.

    div_col_index : int
        Column index of division_id in a.demographics (0-based).

    base_dir : str
        Directory where the three CSVs live.

    N_obs : int
        Pseudo micro sample size (only rescales micro SEs).
    """
    purchase_df, ev_df, vt_df = _load_division_targets(base_dir)

    prod_years  = set(np.unique(product_data["market_ids"]))
    agent_years = set(np.unique(agent_data["market_ids"]))

    purchase_years = prod_years & agent_years & set(purchase_df["market_ids"])
    ev_years       = prod_years & agent_years & set(ev_df["market_ids"])
    vt_years       = prod_years & agent_years & set(vt_df["market_ids"])

    # Filter to overlapping years only
    purchase_df = purchase_df[purchase_df["market_ids"].isin(purchase_years)].copy()
    ev_df       = ev_df[ev_df["market_ids"].isin(ev_years)].copy()
    vt_df       = vt_df[vt_df["market_ids"].isin(vt_years)].copy()

    all_years = sorted(purchase_years | ev_years | vt_years)

    dataset = pyblp.MicroDataset(
        name="US_division_year_moments",
        observations=N_obs,
        compute_weights=lambda t, p, a: (
            np.ones((a.size, 1 + p.size), dtype=float)
            if t in all_years
            else np.zeros((a.size, 1 + p.size), dtype=float)
        ),
    )

    division_purchase_moments: List[pyblp.MicroMoment] = []
    division_ev_moments: List[pyblp.MicroMoment] = []
    division_type_moments: List[pyblp.MicroMoment] = []

    # ---------- 4.1 Purchase probability: P(purchase | division, year) ----------

    for _, row in purchase_df.iterrows():
        year = int(row["market_ids"])
        div_name = row["division"]
        div_id = int(row["division_id"])
        target = float(row["P_purchase_given_division"])

        num_part = pyblp.MicroPart(
            name=f"[year={year}][purchase] E[1{{div={div_name}}}*1{{j>0}}]",
            dataset=dataset,
            compute_values=lambda t, p, a, year=year, div_id=div_id: (
                np.outer(_division_indicator(a, div_id, div_col_index), _product_vec_inside(p))
                if t == year
                else np.zeros((a.size, 1 + p.size), dtype=float)
            ),
        )

        den_part = pyblp.MicroPart(
            name=f"[year={year}][purchase] E[1{{div={div_name}}}]",
            dataset=dataset,
            compute_values=lambda t, p, a, year=year, div_id=div_id: (
                np.outer(_division_indicator(a, div_id, div_col_index), _product_vec_all(p))
                if t == year
                else np.zeros((a.size, 1 + p.size), dtype=float)
            ),
        )

        mm = pyblp.MicroMoment(
            name=f"P(purchase | div={div_name}, year={year})",
            value=target,
            parts=[num_part, den_part],
            compute_value=_ratio,
            compute_gradient=_ratio_grad,
        )
        division_purchase_moments.append(mm)

    # ---------- 4.2 EV share: P(EV | purchase, division, year) ----------

    for _, row in ev_df.iterrows():
        year = int(row["market_ids"])
        div_name = row["division"]
        div_id = int(row["division_id"])
        target = float(row["P_EV_given_division"])

        num_part = pyblp.MicroPart(
            name=f"[year={year}][ev] E[1{{div={div_name}}}*1{{EV}}*1{{j>0}}]",
            dataset=dataset,
            compute_values=lambda t, p, a, year=year, div_id=div_id: (
                np.outer(_division_indicator(a, div_id, div_col_index), _ev_vec_inside(p))
                if t == year
                else np.zeros((a.size, 1 + p.size), dtype=float)
            ),
        )

        den_part = pyblp.MicroPart(
            name=f"[year={year}][ev] E[1{{div={div_name}}}*1{{j>0}}]",
            dataset=dataset,
            compute_values=lambda t, p, a, year=year, div_id=div_id: (
                np.outer(_division_indicator(a, div_id, div_col_index), _product_vec_inside(p))
                if t == year
                else np.zeros((a.size, 1 + p.size), dtype=float)
            ),
        )

        mm = pyblp.MicroMoment(
            name=f"P(EV | purchase, div={div_name}, year={year})",
            value=target,
            parts=[num_part, den_part],
            compute_value=_ratio,
            compute_gradient=_ratio_grad,
        )
        division_ev_moments.append(mm)

    # ---------- 4.3 Vehicle-type share: P(type | purchase, division, year) ----------
    # Keep ONLY Truck and SUV (drop Car and Van)

    for _, row in vt_df.iterrows():
        year    = int(row["market_ids"])
        div_name = row["division"]
        div_id   = int(row["division_id"])
        vt_name  = str(row["vehicle_type"])
        vt_key   = vt_name.strip().upper()

        # keep only Truck and SUV
        if vt_key not in ("TRUCK", "SUV"):
            continue

        target = float(row["P_vehicle_type_given_division"])

        num_part = pyblp.MicroPart(
            name=f"[year={year}][type={vt_name}] E[1{{div={div_name}}}*1{{type={vt_name}}}*1{{j>0}}]",
            dataset=dataset,
            compute_values=lambda t, p, a, year=year, div_id=div_id, vt_name=vt_name: (
                np.outer(_division_indicator(a, div_id, div_col_index), _vt_vec_inside(p, vt_name))
                if t == year
                else np.zeros((a.size, 1 + p.size), dtype=float)
            ),
        )

        den_part = pyblp.MicroPart(
            name=f"[year={year}][type={vt_name}] E[1{{div={div_name}}}*1{{j>0}}]",
            dataset=dataset,
            compute_values=lambda t, p, a, year=year, div_id=div_id: (
                np.outer(_division_indicator(a, div_id, div_col_index), _product_vec_inside(p))
                if t == year
                else np.zeros((a.size, 1 + p.size), dtype=float)
            ),
        )

        mm = pyblp.MicroMoment(
            name=f"P(type={vt_name} | purchase, div={div_name}, year={year})",
            value=target,
            parts=[num_part, den_part],
            compute_value=_ratio,
            compute_gradient=_ratio_grad,
        )
        division_type_moments.append(mm)

    return {
        "purchase": division_purchase_moments,
        "ev": division_ev_moments,
        "types": division_type_moments,
    }


# -------------------------------------------------------------------
# 5. Pooled division moments (over years)
# -------------------------------------------------------------------

def build_division_micro_moments_pooled(
    product_data: pd.DataFrame,
    agent_data: pd.DataFrame,
    div_col_index: int,
    base_dir: str = "moments_output",
    N_obs: int = 1000,
) -> Dict[str, List[pyblp.MicroMoment]]:
    """
    POOLED division moments over multiple years.

    Builds three lists of pooled pyblp.MicroMoment objects:

        'purchase': P(purchase | division)     pooled over years
        'ev':       P(EV | purchase, division) pooled over years
        'types':    P(type | purchase, division) pooled over years

    Pooled targets are simple averages over years of the year-specific
    probabilities in the CSVs, restricted to years present in both
    product_data and agent_data.
    """
    purchase_df, ev_df, vt_df = _load_division_targets(base_dir)

    prod_years  = set(np.unique(product_data["market_ids"]))
    agent_years = set(np.unique(agent_data["market_ids"]))

    # Restrict each table to overlapping years with both product and agent data
    purchase_df = purchase_df[purchase_df["market_ids"].isin(prod_years & agent_years)].copy()
    ev_df       = ev_df[ev_df["market_ids"].isin(prod_years & agent_years)].copy()
    vt_df       = vt_df[vt_df["market_ids"].isin(prod_years & agent_years)].copy()

    purchase_years = set(purchase_df["market_ids"])
    ev_years       = set(ev_df["market_ids"])
    vt_years       = set(vt_df["market_ids"])

    all_years = sorted(purchase_years | ev_years | vt_years)

    # ---- Pooled targets: simple averages over years ----

    # Keep a division name for pretty labels
    if not purchase_df.empty:
        div_names = purchase_df[["division_id", "division"]].drop_duplicates()
    elif not ev_df.empty:
        div_names = ev_df[["division_id", "division"]].drop_duplicates()
    elif not vt_df.empty:
        div_names = vt_df[["division_id", "division"]].drop_duplicates()
    else:
        raise ValueError("No division rows found in any CSVs after year intersection.")

    # P(purchase | division), pooled
    purchase_pooled = (
        purchase_df
        .groupby("division_id", as_index=False)["P_purchase_given_division"]
        .mean()
        .rename(columns={"P_purchase_given_division": "target"})
        .merge(div_names, on="division_id", how="left")
    )

    # P(EV | purchase, division), pooled
    ev_pooled = (
        ev_df
        .groupby("division_id", as_index=False)["P_EV_given_division"]
        .mean()
        .rename(columns={"P_EV_given_division": "target"})
        .merge(div_names, on="division_id", how="left")
    )

    # P(type | purchase, division), pooled (division × vehicle_type)
    vt_pooled = (
        vt_df
        .groupby(["division_id", "vehicle_type"], as_index=False)["P_vehicle_type_given_division"]
        .mean()
        .rename(columns={"P_vehicle_type_given_division": "target"})
        .merge(div_names, on="division_id", how="left")
    )

    # Keep only Truck & SUV for pooled type moments
    vt_pooled["vehicle_type_clean"] = vt_pooled["vehicle_type"].str.strip().str.upper()
    vt_pooled = vt_pooled[vt_pooled["vehicle_type_clean"].isin(["TRUCK", "SUV"])].copy()

    # ---- MicroDataset over all years in which any division moment exists ----
    dataset = pyblp.MicroDataset(
        name="US_division_pooled_moments",
        observations=N_obs,
        compute_weights=lambda t, p, a: (
            np.ones((a.size, 1 + p.size), dtype=float)
            if t in all_years
            else np.zeros((a.size, 1 + p.size), dtype=float)
        ),
    )

    division_purchase_moments: List[pyblp.MicroMoment] = []
    division_ev_moments: List[pyblp.MicroMoment] = []
    division_type_moments: List[pyblp.MicroMoment] = []

    # ---------- 1. P(purchase | division), pooled over years ----------

    for _, row in purchase_pooled.iterrows():
        div_id   = int(row["division_id"])
        div_name = str(row["division"])
        target   = float(row["target"])

        num_part = pyblp.MicroPart(
            name=f"[pooled][purchase] E[1{{div={div_name}}}*1{{j>0}}]",
            dataset=dataset,
            compute_values=lambda t, p, a, div_id=div_id: (
                np.outer(_division_indicator(a, div_id, div_col_index), _product_vec_inside(p))
                if t in purchase_years
                else np.zeros((a.size, 1 + p.size), dtype=float)
            ),
        )

        den_part = pyblp.MicroPart(
            name=f"[pooled][purchase] E[1{{div={div_name}}}]",
            dataset=dataset,
            compute_values=lambda t, p, a, div_id=div_id: (
                np.outer(_division_indicator(a, div_id, div_col_index), _product_vec_all(p))
                if t in purchase_years
                else np.zeros((a.size, 1 + p.size), dtype=float)
            ),
        )

        mm = pyblp.MicroMoment(
            name=f"P(purchase | div={div_name}) pooled",
            value=target,
            parts=[num_part, den_part],
            compute_value=_ratio,
            compute_gradient=_ratio_grad,
        )
        division_purchase_moments.append(mm)

    # ---------- 2. P(EV | purchase, division), pooled over years ----------

    for _, row in ev_pooled.iterrows():
        div_id   = int(row["division_id"])
        div_name = str(row["division"])
        target   = float(row["target"])

        num_part = pyblp.MicroPart(
            name=f"[pooled][ev] E[1{{div={div_name}}}*1{{EV}}*1{{j>0}}]",
            dataset=dataset,
            compute_values=lambda t, p, a, div_id=div_id: (
                np.outer(_division_indicator(a, div_id, div_col_index), _ev_vec_inside(p))
                if t in ev_years
                else np.zeros((a.size, 1 + p.size), dtype=float)
            ),
        )

        den_part = pyblp.MicroPart(
            name=f"[pooled][ev] E[1{{div={div_name}}}*1{{j>0}}]",
            dataset=dataset,
            compute_values=lambda t, p, a, div_id=div_id: (
                np.outer(_division_indicator(a, div_id, div_col_index), _product_vec_inside(p))
                if t in ev_years
                else np.zeros((a.size, 1 + p.size), dtype=float)
            ),
        )

        mm = pyblp.MicroMoment(
            name=f"P(EV | purchase, div={div_name}) pooled",
            value=target,
            parts=[num_part, den_part],
            compute_value=_ratio,
            compute_gradient=_ratio_grad,
        )
        division_ev_moments.append(mm)

    # ---------- 3. P(type | purchase, division), pooled over years ----------
    # Keep ONLY Truck and SUV (vt_pooled already filtered)

    for _, row in vt_pooled.iterrows():
        div_id   = int(row["division_id"])
        div_name = str(row["division"])
        vt_name  = str(row["vehicle_type"])
        target   = float(row["target"])

        num_part = pyblp.MicroPart(
            name=f"[pooled][type={vt_name}] E[1{{div={div_name}}}*1{{type={vt_name}}}*1{{j>0}}]",
            dataset=dataset,
            compute_values=lambda t, p, a, div_id=div_id, vt_name=vt_name: (
                np.outer(_division_indicator(a, div_id, div_col_index), _vt_vec_inside(p, vt_name))
                if t in vt_years
                else np.zeros((a.size, 1 + p.size), dtype=float)
            ),
        )

        den_part = pyblp.MicroPart(
            name=f"[pooled][type={vt_name}] E[1{{div={div_name}}}*1{{j>0}}]",
            dataset=dataset,
            compute_values=lambda t, p, a, div_id=div_id: (
                np.outer(_division_indicator(a, div_id, div_col_index), _product_vec_inside(p))
                if t in vt_years
                else np.zeros((a.size, 1 + p.size), dtype=float)
            ),
        )

        mm = pyblp.MicroMoment(
            name=f"P(type={vt_name} | purchase, div={div_name}) pooled",
            value=target,
            parts=[num_part, den_part],
            compute_value=_ratio,
            compute_gradient=_ratio_grad,
        )
        division_type_moments.append(mm)

    return {
        "purchase": division_purchase_moments,
        "ev":       division_ev_moments,
        "types":    division_type_moments,
    }

def build_division_micro_moments_recent(
    product_data: pd.DataFrame,
    agent_data: pd.DataFrame,
    div_col_index: int,
    base_dir: str = "moments_output",
    N_obs: int = 1000,
    start_year: int = 2021,
    end_year: int = 2024,
) -> Dict[str, List[pyblp.MicroMoment]]:
    """
    Build pooled division moments restricted to a contiguous range of years
    (default: 2021–2024). Targets are simple averages over the selected years.
    """
    purchase_df, ev_df, vt_df = _load_division_targets(base_dir)

    prod_years  = set(np.unique(product_data["market_ids"]))
    agent_years = set(np.unique(agent_data["market_ids"]))
    candidate_years = {y for y in range(start_year, end_year + 1)}

    def restrict(df):
        years = prod_years & agent_years & set(df["market_ids"]) & candidate_years
        return df[df["market_ids"].isin(years)].copy(), years

    purchase_df, purchase_years = restrict(purchase_df)
    ev_df, ev_years = restrict(ev_df)
    vt_df, vt_years = restrict(vt_df)

    all_years = sorted(purchase_years | ev_years | vt_years)
    if not all_years:
        raise ValueError("No overlapping markets found for requested years.")

    def pooled_avg(df, value_col, group_cols):
        if df.empty:
            return pd.DataFrame(columns=group_cols + ["target"])
        pooled = (
            df.groupby(group_cols, as_index=False)[value_col]
            .mean()
            .rename(columns={value_col: "target"})
        )
        name_cols = df[["division_id", "division"]].drop_duplicates()
        return pooled.merge(name_cols, on="division_id", how="left")

    purchase_pooled = pooled_avg(purchase_df, "P_purchase_given_division", ["division_id"])
    ev_pooled = pooled_avg(ev_df, "P_EV_given_division", ["division_id"])
    vt_pooled = pooled_avg(
        vt_df,
        "P_vehicle_type_given_division",
        ["division_id", "vehicle_type"],
    )
    vt_pooled["vehicle_type_clean"] = vt_pooled["vehicle_type"].str.strip().str.upper()
    vt_pooled = vt_pooled[vt_pooled["vehicle_type_clean"].isin(["TRUCK", "SUV"])].copy()

    dataset = pyblp.MicroDataset(
        name=f"US_division_{start_year}_{end_year}_moments",
        observations=N_obs,
        compute_weights=lambda t, p, a: (
            np.ones((a.size, 1 + p.size), dtype=float)
            if t in all_years
            else np.zeros((a.size, 1 + p.size), dtype=float)
        ),
    )

    division_purchase_moments: List[pyblp.MicroMoment] = []
    division_ev_moments: List[pyblp.MicroMoment] = []
    division_type_moments: List[pyblp.MicroMoment] = []

    for _, row in purchase_pooled.iterrows():
        div_id = int(row["division_id"])
        div_name = str(row["division"])
        target = float(row["target"])

        num_part = pyblp.MicroPart(
            name=f"[{start_year}-{end_year}][purchase] E[1{{div={div_name}}}*1{{j>0}}]",
            dataset=dataset,
            compute_values=lambda t, p, a, div_id=div_id: (
                np.outer(_division_indicator(a, div_id, div_col_index), _product_vec_inside(p))
                if t in purchase_years
                else np.zeros((a.size, 1 + p.size), dtype=float)
            ),
        )

        den_part = pyblp.MicroPart(
            name=f"[{start_year}-{end_year}][purchase] E[1{{div={div_name}}}]",
            dataset=dataset,
            compute_values=lambda t, p, a, div_id=div_id: (
                np.outer(_division_indicator(a, div_id, div_col_index), _product_vec_all(p))
                if t in purchase_years
                else np.zeros((a.size, 1 + p.size), dtype=float)
            ),
        )

        division_purchase_moments.append(
            pyblp.MicroMoment(
                name=f"P(purchase | div={div_name}) {start_year}-{end_year}",
                value=target,
                parts=[num_part, den_part],
                compute_value=_ratio,
                compute_gradient=_ratio_grad,
            )
        )

    for _, row in ev_pooled.iterrows():
        div_id = int(row["division_id"])
        div_name = str(row["division"])
        target = float(row["target"])

        num_part = pyblp.MicroPart(
            name=f"[{start_year}-{end_year}][ev] E[1{{div={div_name}}}*1{{EV}}*1{{j>0}}]",
            dataset=dataset,
            compute_values=lambda t, p, a, div_id=div_id: (
                np.outer(_division_indicator(a, div_id, div_col_index), _ev_vec_inside(p))
                if t in ev_years
                else np.zeros((a.size, 1 + p.size), dtype=float)
            ),
        )

        den_part = pyblp.MicroPart(
            name=f"[{start_year}-{end_year}][ev] E[1{{div={div_name}}}*1{{j>0}}]",
            dataset=dataset,
            compute_values=lambda t, p, a, div_id=div_id: (
                np.outer(_division_indicator(a, div_id, div_col_index), _product_vec_inside(p))
                if t in ev_years
                else np.zeros((a.size, 1 + p.size), dtype=float)
            ),
        )

        division_ev_moments.append(
            pyblp.MicroMoment(
                name=f"P(EV | purchase, div={div_name}) {start_year}-{end_year}",
                value=target,
                parts=[num_part, den_part],
                compute_value=_ratio,
                compute_gradient=_ratio_grad,
            )
        )

    for _, row in vt_pooled.iterrows():
        div_id = int(row["division_id"])
        div_name = str(row["division"])
        vt_name = str(row["vehicle_type"])
        target = float(row["target"])

        num_part = pyblp.MicroPart(
            name=f"[{start_year}-{end_year}][type={vt_name}] E[1{{div={div_name}}}*1{{type={vt_name}}}*1{{j>0}}]",
            dataset=dataset,
            compute_values=lambda t, p, a, div_id=div_id, vt_name=vt_name: (
                np.outer(_division_indicator(a, div_id, div_col_index), _vt_vec_inside(p, vt_name))
                if t in vt_years
                else np.zeros((a.size, 1 + p.size), dtype=float)
            ),
        )

        den_part = pyblp.MicroPart(
            name=f"[{start_year}-{end_year}][type={vt_name}] E[1{{div={div_name}}}*1{{j>0}}]",
            dataset=dataset,
            compute_values=lambda t, p, a, div_id=div_id: (
                np.outer(_division_indicator(a, div_id, div_col_index), _product_vec_inside(p))
                if t in vt_years
                else np.zeros((a.size, 1 + p.size), dtype=float)
            ),
        )

        division_type_moments.append(
            pyblp.MicroMoment(
                name=f"P(type={vt_name} | purchase, div={div_name}) {start_year}-{end_year}",
                value=target,
                parts=[num_part, den_part],
                compute_value=_ratio,
                compute_gradient=_ratio_grad,
            )
        )

    return {
        "purchase": division_purchase_moments,
        "ev": division_ev_moments,
        "types": division_type_moments,
    }


# -------------------------------------------------------------------
# 6. Single-year (e.g., 2024) division moments
# -------------------------------------------------------------------

def build_division_micro_moments_2024(
    product_data: pd.DataFrame,
    agent_data: pd.DataFrame,
    div_col_index: int,
    base_dir: str = "moments_output",
    N_obs: int = 1000,
    target_year: int = 2024,
) -> Dict[str, List[pyblp.MicroMoment]]:
    """
    Build three lists of pyblp.MicroMoment objects for a *single* year
    (default: 2024):

        'purchase': P(purchase | division, year=target_year)
        'ev':       P(EV | purchase, division, year=target_year)
        'types':    P(vehicle_type | purchase, division, year=target_year)
    """
    purchase_df, ev_df, vt_df = _load_division_targets(base_dir)

    prod_years  = set(np.unique(product_data["market_ids"]))
    agent_years = set(np.unique(agent_data["market_ids"]))

    # Check that target_year is present in both product and agent data
    if target_year not in prod_years:
        raise ValueError(f"target_year={target_year} not found in product_data['market_ids'].")
    if target_year not in agent_years:
        raise ValueError(f"target_year={target_year} not found in agent_data['market_ids'].")

    # Restrict CSVs to just the target_year
    purchase_df = purchase_df[purchase_df["market_ids"] == target_year].copy()
    ev_df       = ev_df[ev_df["market_ids"] == target_year].copy()
    vt_df       = vt_df[vt_df["market_ids"] == target_year].copy()

    if purchase_df.empty and ev_df.empty and vt_df.empty:
        raise ValueError(f"No division rows found in any CSVs for market_ids={target_year}.")

    # Weights only active in the target_year
    dataset = pyblp.MicroDataset(
        name=f"US_division_year_{target_year}_moments",
        observations=N_obs,
        compute_weights=lambda t, p, a, _ty=target_year: (
            np.ones((a.size, 1 + p.size), dtype=float)
            if t == _ty
            else np.zeros((a.size, 1 + p.size), dtype=float)
        ),
    )

    division_purchase_moments: List[pyblp.MicroMoment] = []
    division_ev_moments: List[pyblp.MicroMoment] = []
    division_type_moments: List[pyblp.MicroMoment] = []

    # ---------- 1. P(purchase | division, target_year) ----------

    for _, row in purchase_df.iterrows():
        year    = int(row["market_ids"])  # will equal target_year
        div_name = row["division"]
        div_id   = int(row["division_id"])
        target   = float(row["P_purchase_given_division"])

        num_part = pyblp.MicroPart(
            name=f"[year={year}][purchase] E[1{{div={div_name}}}*1{{j>0}}]",
            dataset=dataset,
            compute_values=lambda t, p, a, year=year, div_id=div_id: (
                np.outer(_division_indicator(a, div_id, div_col_index), _product_vec_inside(p))
                if t == year
                else np.zeros((a.size, 1 + p.size), dtype=float)
            ),
        )

        den_part = pyblp.MicroPart(
            name=f"[year={year}][purchase] E[1{{div={div_name}}}]",
            dataset=dataset,
            compute_values=lambda t, p, a, year=year, div_id=div_id: (
                np.outer(_division_indicator(a, div_id, div_col_index), _product_vec_all(p))
                if t == year
                else np.zeros((a.size, 1 + p.size), dtype=float)
            ),
        )

        mm = pyblp.MicroMoment(
            name=f"P(purchase | div={div_name}, year={year})",
            value=target,
            parts=[num_part, den_part],
            compute_value=_ratio,
            compute_gradient=_ratio_grad,
        )
        division_purchase_moments.append(mm)

    # ---------- 2. P(EV | purchase, division, target_year) ----------

    for _, row in ev_df.iterrows():
        year    = int(row["market_ids"])  # = target_year
        div_name = row["division"]
        div_id   = int(row["division_id"])
        target   = float(row["P_EV_given_division"])

        num_part = pyblp.MicroPart(
            name=f"[year={year}][ev] E[1{{div={div_name}}}*1{{EV}}*1{{j>0}}]",
            dataset=dataset,
            compute_values=lambda t, p, a, year=year, div_id=div_id: (
                np.outer(_division_indicator(a, div_id, div_col_index), _ev_vec_inside(p))
                if t == year
                else np.zeros((a.size, 1 + p.size), dtype=float)
            ),
        )

        den_part = pyblp.MicroPart(
            name=f"[year={year}][ev] E[1{{div={div_name}}}*1{{j>0}}]",
            dataset=dataset,
            compute_values=lambda t, p, a, year=year, div_id=div_id: (
                np.outer(_division_indicator(a, div_id, div_col_index), _product_vec_inside(p))
                if t == year
                else np.zeros((a.size, 1 + p.size), dtype=float)
            ),
        )

        mm = pyblp.MicroMoment(
            name=f"P(EV | purchase, div={div_name}, year={year})",
            value=target,
            parts=[num_part, den_part],
            compute_value=_ratio,
            compute_gradient=_ratio_grad,
        )
        division_ev_moments.append(mm)

    # ---------- 3. P(type | purchase, division, target_year) ----------
    # Keep ONLY Truck and SUV

    for _, row in vt_df.iterrows():
        year    = int(row["market_ids"])  # = target_year
        div_name = row["division"]
        div_id   = int(row["division_id"])
        vt_name  = str(row["vehicle_type"])
        vt_key   = vt_name.strip().upper()

        if vt_key not in ("TRUCK", "SUV"):
            continue

        target   = float(row["P_vehicle_type_given_division"])

        num_part = pyblp.MicroPart(
            name=f"[year={year}][type={vt_name}] E[1{{div={div_name}}}*1{{type={vt_name}}}*1{{j>0}}]",
            dataset=dataset,
            compute_values=lambda t, p, a, year=year, div_id=div_id, vt_name=vt_name: (
                np.outer(_division_indicator(a, div_id, div_col_index), _vt_vec_inside(p, vt_name))
                if t == year
                else np.zeros((a.size, 1 + p.size), dtype=float)
            ),
        )

        den_part = pyblp.MicroPart(
            name=f"[year={year}][type={vt_name}] E[1{{div={div_name}}}*1{{j>0}}]",
            dataset=dataset,
            compute_values=lambda t, p, a, year=year, div_id=div_id: (
                np.outer(_division_indicator(a, div_id, div_col_index), _product_vec_inside(p))
                if t == year
                else np.zeros((a.size, 1 + p.size), dtype=float)
            ),
        )

        mm = pyblp.MicroMoment(
            name=f"P(type={vt_name} | purchase, div={div_name}, year={year})",
            value=target,
            parts=[num_part, den_part],
            compute_value=_ratio,
            compute_gradient=_ratio_grad,
        )
        division_type_moments.append(mm)

    return {
        "purchase": division_purchase_moments,
        "ev":       division_ev_moments,
        "types":    division_type_moments,
    }


# -------------------------------------------------------------------
# 7. Pacific-only division moments (across years)
# -------------------------------------------------------------------

def build_division_micro_moments_pacific_only(
    product_data: pd.DataFrame,
    agent_data: pd.DataFrame,
    div_col_index: int,
    base_dir: str = "moments_output",
    N_obs: int = 1000,
) -> Dict[str, List[pyblp.MicroMoment]]:
    """
    Build three lists of pyblp.MicroMoment objects, but ONLY for the Pacific division
    (division_id = 9), across all years where we have both product/agent data and
    Pacific targets in the CSVs:

        'purchase': P(purchase | division=Pacific, year)
        'ev':       P(EV | purchase, division=Pacific, year)
        'types':    P(vehicle_type | purchase, division=Pacific, year)
    """
    purchase_df, ev_df, vt_df = _load_division_targets(base_dir)

    # Restrict to Pacific division only
    pacific_name = "Pacific"
    pacific_id = DIVISION_NAME_TO_ID[pacific_name]

    purchase_df = purchase_df[purchase_df["division_id"] == pacific_id].copy()
    ev_df       = ev_df[ev_df["division_id"] == pacific_id].copy()
    vt_df       = vt_df[vt_df["division_id"] == pacific_id].copy()

    # Years present in product/agent data
    prod_years  = set(np.unique(product_data["market_ids"]))
    agent_years = set(np.unique(agent_data["market_ids"]))

    purchase_years = prod_years & agent_years & set(purchase_df["market_ids"])
    ev_years       = prod_years & agent_years & set(ev_df["market_ids"])
    vt_years       = prod_years & agent_years & set(vt_df["market_ids"])

    # Filter to overlapping years only
    purchase_df = purchase_df[purchase_df["market_ids"].isin(purchase_years)].copy()
    ev_df       = ev_df[ev_df["market_ids"].isin(ev_years)].copy()
    vt_df       = vt_df[vt_df["market_ids"].isin(vt_years)].copy()

    all_years = sorted(purchase_years | ev_years | vt_years)

    if not all_years:
        raise ValueError("No overlapping years with Pacific division moments found.")

    dataset = pyblp.MicroDataset(
        name="US_pacific_division_year_moments",
        observations=N_obs,
        compute_weights=lambda t, p, a: (
            np.ones((a.size, 1 + p.size), dtype=float)
            if t in all_years
            else np.zeros((a.size, 1 + p.size), dtype=float)
        ),
    )

    division_purchase_moments: List[pyblp.MicroMoment] = []
    division_ev_moments: List[pyblp.MicroMoment] = []
    division_type_moments: List[pyblp.MicroMoment] = []

    # ---------- 1. P(purchase | Pacific, year) ----------

    for _, row in purchase_df.iterrows():
        year     = int(row["market_ids"])
        div_name = row["division"]        # should be "Pacific"
        div_id   = int(row["division_id"])  # should be 9
        target   = float(row["P_purchase_given_division"])

        num_part = pyblp.MicroPart(
            name=f"[year={year}][purchase] E[1{{div={div_name}}}*1{{j>0}}]",
            dataset=dataset,
            compute_values=lambda t, p, a, year=year, div_id=div_id: (
                np.outer(_division_indicator(a, div_id, div_col_index), _product_vec_inside(p))
                if t == year
                else np.zeros((a.size, 1 + p.size), dtype=float)
            ),
        )

        den_part = pyblp.MicroPart(
            name=f"[year={year}][purchase] E[1{{div={div_name}}}]",
            dataset=dataset,
            compute_values=lambda t, p, a, year=year, div_id=div_id: (
                np.outer(_division_indicator(a, div_id, div_col_index), _product_vec_all(p))
                if t == year
                else np.zeros((a.size, 1 + p.size), dtype=float)
            ),
        )

        mm = pyblp.MicroMoment(
            name=f"P(purchase | div={div_name}, year={year})",
            value=target,
            parts=[num_part, den_part],
            compute_value=_ratio,
            compute_gradient=_ratio_grad,
        )
        division_purchase_moments.append(mm)

    # ---------- 2. P(EV | purchase, Pacific, year) ----------

    for _, row in ev_df.iterrows():
        year     = int(row["market_ids"])
        div_name = row["division"]
        div_id   = int(row["division_id"])
        target   = float(row["P_EV_given_division"])

        num_part = pyblp.MicroPart(
            name=f"[year={year}][ev] E[1{{div={div_name}}}*1{{EV}}*1{{j>0}}]",
            dataset=dataset,
            compute_values=lambda t, p, a, year=year, div_id=div_id: (
                np.outer(_division_indicator(a, div_id, div_col_index), _ev_vec_inside(p))
                if t == year
                else np.zeros((a.size, 1 + p.size), dtype=float)
            ),
        )

        den_part = pyblp.MicroPart(
            name=f"[year={year}][ev] E[1{{div={div_name}}}*1{{j>0}}]",
            dataset=dataset,
            compute_values=lambda t, p, a, year=year, div_id=div_id: (
                np.outer(_division_indicator(a, div_id, div_col_index), _product_vec_inside(p))
                if t == year
                else np.zeros((a.size, 1 + p.size), dtype=float)
            ),
        )

        mm = pyblp.MicroMoment(
            name=f"P(EV | purchase, div={div_name}, year={year})",
            value=target,
            parts=[num_part, den_part],
            compute_value=_ratio,
            compute_gradient=_ratio_grad,
        )
        division_ev_moments.append(mm)

    # ---------- 3. P(type | purchase, Pacific, year) ----------
    # Keep ONLY Truck and SUV

    for _, row in vt_df.iterrows():
        year     = int(row["market_ids"])
        div_name = row["division"]
        div_id   = int(row["division_id"])
        vt_name  = str(row["vehicle_type"])
        vt_key   = vt_name.strip().upper()

        if vt_key not in ("TRUCK", "SUV"):
            continue

        target   = float(row["P_vehicle_type_given_division"])

        num_part = pyblp.MicroPart(
            name=f"[year={year}][type={vt_name}] E[1{{div={div_name}}}*1{{type={vt_name}}}*1{{j>0}}]",
            dataset=dataset,
            compute_values=lambda t, p, a, year=year, div_id=div_id, vt_name=vt_name: (
                np.outer(_division_indicator(a, div_id, div_col_index), _vt_vec_inside(p, vt_name))
                if t == year
                else np.zeros((a.size, 1 + p.size), dtype=float)
            ),
        )

        den_part = pyblp.MicroPart(
            name=f"[year={year}][type={vt_name}] E[1{{div={div_name}}}*1{{j>0}}]",
            dataset=dataset,
            compute_values=lambda t, p, a, year=year, div_id=div_id: (
                np.outer(_division_indicator(a, div_id, div_col_index), _product_vec_inside(p))
                if t == year
                else np.zeros((a.size, 1 + p.size), dtype=float)
            ),
        )

        mm = pyblp.MicroMoment(
            name=f"P(type={vt_name} | purchase, div={div_name}, year={year})",
            value=target,
            parts=[num_part, den_part],
            compute_value=_ratio,
            compute_gradient=_ratio_grad,
        )
        division_type_moments.append(mm)

    return {
        "purchase": division_purchase_moments,
        "ev":       division_ev_moments,
        "types":    division_type_moments,
    }
