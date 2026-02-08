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
    "North East":     1,
    "North Central":  2,
    "South Atlantic": 3,
    "South Central":  4,
    "Mountain":       5,
    "Pacific":        6,
}

# Mapping from original 9 Census division names (in CSV files) to
# condensed 6-division labels used in DIVISION_NAME_TO_ID.
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
    """Safely compute num / den for a 2-vector v = [num, den]."""
    num, den = v[0], v[1] + 1e-16
    return float(num / den)


def _ratio_grad(v: np.ndarray) -> np.ndarray:
    """Gradient of num/den with respect to (num, den)."""
    num, den = v[0], v[1] + 1e-16
    return np.array([1.0 / den, -num / (den ** 2)], dtype=float)


def _division_indicator(
    a: any,
    division_id: int,
    first_div_col: int,
    n_divisions: int | None = None,
) -> np.ndarray:
    """
    Agent-side indicator 1{division == division_id} when demographics contain
    division dummies [div_1, ..., div_n] starting at column first_div_col.

    To avoid repeatedly slicing demographics, we cache the block for each
    (first_div_col, n_divisions) pair on the agents object.
    """
    if n_divisions is None:
        n_divisions = len(DIVISION_NAME_TO_ID)

    cache_key = (first_div_col, n_divisions)
    cache_attr = "_division_indicator_cache"
    cache = getattr(a, cache_attr, None)

    if cache is None or cache.get("key") != cache_key:
        div_block = a.demographics[:, first_div_col:first_div_col + n_divisions]
        cache = {"key": cache_key, "block": div_block, "vectors": {}}
        setattr(a, cache_attr, cache)

    vectors = cache["vectors"]
    if division_id not in vectors:
        # division_id is 1-based, columns are 0-based
        vectors[division_id] = cache["block"][:, division_id - 1]

    return vectors[division_id]


def _product_vec_inside(p) -> np.ndarray:
    """[0, 1, 1, ..., 1] – inside options only (j > 0)."""
    return np.r_[0.0, np.ones(p.size, dtype=float)]


def _product_vec_all(p) -> np.ndarray:
    """[1, 1, 1, ..., 1] – outside + inside."""
    return np.r_[1.0, np.ones(p.size, dtype=float)]


def _ev_vec_inside(p) -> np.ndarray:
    """[0, EV_1, ..., EV_J] – EV dummy, outside = 0."""
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

def _standardize_division_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Harmonize a division-level CSV so that it has:
      - 'market_ids' column for the year
      - 'division' using condensed 6-division labels
      - 'division_id' integer 1..6 matching DIVISION_NAME_TO_ID
    Works whether the CSV stores 9 original Census names or 6 condensed names.
    """
    df = df.copy()

    # Standardize year column
    if "market_year" in df.columns:
        df = df.rename(columns={"market_year": "market_ids"})

    # Map any 9-division labels down to 6-division labels; keep already
    # condensed labels as-is.
    df["division"] = df["division"].map(
        lambda x: CENSUS_DIV_TO_CONDENSED.get(x, x)
    )

    # Map division name -> numeric id
    df["division_id"] = df["division"].map(DIVISION_NAME_TO_ID)

    # Drop rows that don't map cleanly
    df = df.dropna(subset=["division_id"]).copy()
    df["division_id"] = df["division_id"].astype(int)

    return df


def _load_division_targets(base_dir: str):
    """Return three dataframes with year-division targets."""
    purchase = pd.read_csv(os.path.join(base_dir, "prob_purchase_given_division.csv"))
    ev       = pd.read_csv(os.path.join(base_dir, "prob_EV_given_division.csv"))
    vt       = pd.read_csv(os.path.join(base_dir, "prob_vehicle_type_given_division.csv"))

    purchase = _standardize_division_df(purchase)
    ev       = _standardize_division_df(ev)
    vt       = _standardize_division_df(vt)

    return purchase, ev, vt


# -------------------------------------------------------------------
# 4. Pooled division moments over *all* overlapping years
# -------------------------------------------------------------------

def build_division_micro_moments_pooled(
    product_data: pd.DataFrame,
    agent_data: pd.DataFrame,
    div_col_index: int,
    base_dir: str = "moments_output",
    N_obs: int = 1000,
    n_divisions: int | None = None,
) -> Dict[str, List[pyblp.MicroMoment]]:
    """
    POOLED division moments over multiple years.

    Builds three lists of pooled pyblp.MicroMoment objects:

        'purchase': P(purchase | division)          pooled over years
        'ev':       P(EV | purchase, division)      pooled over years
        'types':    P(type | purchase, division)    pooled over years

    Pooled targets are simple averages over years of the year-specific
    probabilities in the CSVs, restricted to years present in both
    product_data and agent_data.
    """
    purchase_df, ev_df, vt_df = _load_division_targets(base_dir)

    if n_divisions is None:
        n_divisions = len(DIVISION_NAME_TO_ID)

    prod_years  = set(np.unique(product_data["market_ids"]))
    agent_years = set(np.unique(agent_data["market_ids"]))

    # Restrict each table to overlapping years with both product and agent data
    purchase_df = purchase_df[purchase_df["market_ids"].isin(prod_years & agent_years)].copy()
    ev_df       = ev_df[ev_df["market_ids"].isin(prod_years & agent_years)].copy()
    vt_df       = vt_df[vt_df["market_ids"].isin(prod_years & agent_years)].copy()

    purchase_years = set(purchase_df["market_ids"])
    ev_years       = set(ev_df["market_ids"])
    vt_years       = set(vt_df["market_ids"])
    all_years      = sorted(purchase_years | ev_years | vt_years)

    if not all_years:
        raise ValueError("No overlapping years in product/agent/division CSVs for pooled division moments.")

    # ---- Pooled targets: simple averages over years ----
    if not purchase_df.empty:
        div_names = purchase_df[["division_id", "division"]].drop_duplicates()
    elif not ev_df.empty:
        div_names = ev_df[["division_id", "division"]].drop_duplicates()
    elif not vt_df.empty:
        div_names = vt_df[["division_id", "division"]].drop_duplicates()
    else:
        raise ValueError("No division rows found in any CSVs after year intersection.")

    def pooled_avg(df: pd.DataFrame, value_col: str, group_cols: list[str]) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=group_cols + ["target", "division"])
        pooled = (
            df.groupby(group_cols, as_index=False)[value_col]
            .mean()
            .rename(columns={value_col: "target"})
        )
        return pooled.merge(div_names, on="division_id", how="left")

    purchase_pooled = pooled_avg(purchase_df, "P_purchase_given_division", ["division_id"])
    ev_pooled       = pooled_avg(ev_df,       "P_EV_given_division",       ["division_id"])
    vt_pooled       = pooled_avg(vt_df,       "P_vehicle_type_given_division", ["division_id", "vehicle_type"])

    # MicroDataset over all overlapping years
    dataset = pyblp.MicroDataset(
        name="US_division_pooled_moments",
        observations=N_obs,
        market_ids=all_years,
        compute_weights=lambda t, p, a: np.ones((a.size, 1 + p.size), dtype=float),
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
            compute_values=lambda t, p, a, div_id=div_id: np.outer(
                _division_indicator(a, div_id, div_col_index, n_divisions),
                _product_vec_inside(p),
            ),
        )

        den_part = pyblp.MicroPart(
            name=f"[pooled][purchase] E[1{{div={div_name}}}]",
            dataset=dataset,
            compute_values=lambda t, p, a, div_id=div_id: np.outer(
                _division_indicator(a, div_id, div_col_index, n_divisions),
                _product_vec_all(p),
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
            compute_values=lambda t, p, a, div_id=div_id: np.outer(
                _division_indicator(a, div_id, div_col_index, n_divisions),
                _ev_vec_inside(p),
            ),
        )

        den_part = pyblp.MicroPart(
            name=f"[pooled][ev] E[1{{div={div_name}}}*1{{j>0}}]",
            dataset=dataset,
            compute_values=lambda t, p, a, div_id=div_id: np.outer(
                _division_indicator(a, div_id, div_col_index, n_divisions),
                _product_vec_inside(p),
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

    for _, row in vt_pooled.iterrows():
        div_id   = int(row["division_id"])
        div_name = str(row["division"])
        vt_name  = str(row["vehicle_type"])
        target   = float(row["target"])

        num_part = pyblp.MicroPart(
            name=f"[pooled][type={vt_name}] E[1{{div={div_name}}}*1{{type={vt_name}}}*1{{j>0}}]",
            dataset=dataset,
            compute_values=lambda t, p, a, div_id=div_id, vt_name=vt_name: np.outer(
                _division_indicator(a, div_id, div_col_index, n_divisions),
                _vt_vec_inside(p, vt_name),
            ),
        )

        den_part = pyblp.MicroPart(
            name=f"[pooled][type={vt_name}] E[1{{div={div_name}}}*1{{j>0}}]",
            dataset=dataset,
            compute_values=lambda t, p, a, div_id=div_id: np.outer(
                _division_indicator(a, div_id, div_col_index, n_divisions),
                _product_vec_inside(p),
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


# -------------------------------------------------------------------
# 5. YEAR-SPECIFIC division moments for a range of years (e.g., 2021–2024)
# -------------------------------------------------------------------

def build_division_micro_moments_yearly_range(
    product_data: pd.DataFrame,
    agent_data: pd.DataFrame,
    div_col_index: int,
    base_dir: str = "moments_output",
    N_obs: int = 1000,
    start_year: int = 2021,
    end_year: int = 2024,
    n_divisions: int | None = None,
) -> Dict[str, List[pyblp.MicroMoment]]:
    """
    Build YEAR-SPECIFIC division micro moments over a contiguous range of years.

    For each year t in [start_year, end_year] that appears in both product_data
    and agent_data and in the division CSVs, we construct:

        'purchase': P(purchase | division, year=t)
        'ev':       P(EV | purchase, division, year=t)
        'types':    P(type | purchase, division, year=t)

    Additionally, in the returned dictionary we include two convenience lists:

        'types_truck': only the truck moments (type == 'Truck')
        'types_suv':   only the SUV moments   (type == 'SUV')

    The MicroDataset backing these moments is explicitly restricted to the
    relevant years via the market_ids argument, mirroring how SecondChoice2015
    micro moments are limited to a single market_id.
    """
    purchase_df, ev_df, vt_df = _load_division_targets(base_dir)

    if n_divisions is None:
        n_divisions = len(DIVISION_NAME_TO_ID)

    prod_years  = set(np.unique(product_data["market_ids"]))
    agent_years = set(np.unique(agent_data["market_ids"]))
    year_window = set(range(start_year, end_year + 1))

    purchase_years = year_window & prod_years & agent_years & set(purchase_df["market_ids"])
    ev_years       = year_window & prod_years & agent_years & set(ev_df["market_ids"])
    vt_years       = year_window & prod_years & agent_years & set(vt_df["market_ids"])

    # Filter each dataframe down to the years it will actually use
    purchase_df = purchase_df[purchase_df["market_ids"].isin(purchase_years)].copy()
    ev_df       = ev_df[ev_df["market_ids"].isin(ev_years)].copy()
    vt_df       = vt_df[vt_df["market_ids"].isin(vt_years)].copy()

    all_years = sorted(purchase_years | ev_years | vt_years)

    if not all_years:
        raise ValueError(
            f"No overlapping years in [{start_year}, {end_year}] between product_data, "
            f"agent_data, and division CSVs."
        )

    # MicroDataset that *only* lives on the requested years
    dataset = pyblp.MicroDataset(
        name=f"US_division_{start_year}_{end_year}_yearly_moments",
        observations=N_obs,
        market_ids=all_years,
        compute_weights=lambda t, p, a: np.ones((a.size, 1 + p.size), dtype=float),
    )

    division_purchase_moments: List[pyblp.MicroMoment] = []
    division_ev_moments: List[pyblp.MicroMoment] = []
    division_type_moments: List[pyblp.MicroMoment] = []
    division_truck_moments: List[pyblp.MicroMoment] = []
    division_suv_moments: List[pyblp.MicroMoment] = []

    # ---------- 1. P(purchase | division, year) ----------

    for _, row in purchase_df.iterrows():
        year     = int(row["market_ids"])
        div_name = str(row["division"])
        div_id   = int(row["division_id"])
        target   = float(row["P_purchase_given_division"])

        num_part = pyblp.MicroPart(
            name=f"[year={year}][purchase] E[1{{div={div_name}}}*1{{j>0}}]",
            dataset=dataset,
            compute_values=lambda t, p, a, year=year, div_id=div_id: (
                np.outer(
                    _division_indicator(a, div_id, div_col_index, n_divisions),
                    _product_vec_inside(p),
                )
                if t == year
                else np.zeros((a.size, 1 + p.size), dtype=float)
            ),
        )

        den_part = pyblp.MicroPart(
            name=f"[year={year}][purchase] E[1{{div={div_name}}}]",
            dataset=dataset,
            compute_values=lambda t, p, a, year=year, div_id=div_id: (
                np.outer(
                    _division_indicator(a, div_id, div_col_index, n_divisions),
                    _product_vec_all(p),
                )
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

    # ---------- 2. P(EV | purchase, division, year) ----------

    for _, row in ev_df.iterrows():
        year     = int(row["market_ids"])
        div_name = str(row["division"])
        div_id   = int(row["division_id"])
        target   = float(row["P_EV_given_division"])

        num_part = pyblp.MicroPart(
            name=f"[year={year}][ev] E[1{{div={div_name}}}*1{{EV}}*1{{j>0}}]",
            dataset=dataset,
            compute_values=lambda t, p, a, year=year, div_id=div_id: (
                np.outer(
                    _division_indicator(a, div_id, div_col_index, n_divisions),
                    _ev_vec_inside(p),
                )
                if t == year
                else np.zeros((a.size, 1 + p.size), dtype=float)
            ),
        )

        den_part = pyblp.MicroPart(
            name=f"[year={year}][ev] E[1{{div={div_name}}}*1{{j>0}}]",
            dataset=dataset,
            compute_values=lambda t, p, a, year=year, div_id=div_id: (
                np.outer(
                    _division_indicator(a, div_id, div_col_index, n_divisions),
                    _product_vec_inside(p),
                )
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

    # ---------- 3. P(type | purchase, division, year) ----------

    for _, row in vt_df.iterrows():
        year     = int(row["market_ids"])
        div_name = str(row["division"])
        div_id   = int(row["division_id"])
        vt_name  = str(row["vehicle_type"])
        target   = float(row["P_vehicle_type_given_division"])

        num_part = pyblp.MicroPart(
            name=f"[year={year}][type={vt_name}] E[1{{div={div_name}}}*1{{type={vt_name}}}*1{{j>0}}]",
            dataset=dataset,
            compute_values=lambda t, p, a, year=year, div_id=div_id, vt_name=vt_name: (
                np.outer(
                    _division_indicator(a, div_id, div_col_index, n_divisions),
                    _vt_vec_inside(p, vt_name),
                )
                if t == year
                else np.zeros((a.size, 1 + p.size), dtype=float)
            ),
        )

        den_part = pyblp.MicroPart(
            name=f"[year={year}][type={vt_name}] E[1{{div={div_name}}}*1{{j>0}}]",
            dataset=dataset,
            compute_values=lambda t, p, a, year=year, div_id=div_id: (
                np.outer(
                    _division_indicator(a, div_id, div_col_index, n_divisions),
                    _product_vec_inside(p),
                )
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

        vt_upper = vt_name.strip().upper()
        if vt_upper == "TRUCK":
            division_truck_moments.append(mm)
        elif vt_upper == "SUV":
            division_suv_moments.append(mm)

    return {
        "purchase":     division_purchase_moments,
        "ev":           division_ev_moments,
        "types":        division_type_moments,
        "types_truck":  division_truck_moments,
        "types_suv":    division_suv_moments,
    }


# -------------------------------------------------------------------
# 6. Convenience wrapper for a *single* year (e.g. 2024 only)
# -------------------------------------------------------------------

def build_division_micro_moments_2024(
    product_data: pd.DataFrame,
    agent_data: pd.DataFrame,
    div_col_index: int,
    base_dir: str = "moments_output",
    N_obs: int = 1000,
) -> Dict[str, List[pyblp.MicroMoment]]:
    """
    Backwards-compatible wrapper for the single-year case.

    This returns the same structure as build_division_micro_moments_yearly_range,
    but restricted to year=2024. For compatibility with earlier code, only the
    'purchase', 'ev', and 'types' lists are returned.
    """
    all_moments = build_division_micro_moments_yearly_range(
        product_data=product_data,
        agent_data=agent_data,
        div_col_index=div_col_index,
        base_dir=base_dir,
        N_obs=N_obs,
        start_year=2024,
        end_year=2024,
    )

    return {
        "purchase": all_moments["purchase"],
        "ev":       all_moments["ev"],
        "types":    all_moments["types"],
    }
