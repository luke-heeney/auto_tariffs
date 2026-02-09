from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pyblp


def _norm(x) -> str:
    return str(x).strip().lower()


def load_owner_map(path: str | Path, *, sheet: str = "brands", owner_col: str = "owner") -> dict[str, str]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Owner mapping file not found: {path}")
    df = pd.read_excel(path, sheet_name=sheet)
    cols = {c.strip().lower(): c for c in df.columns}
    owner_key = owner_col.strip().lower()
    if "brand" not in cols or owner_key not in cols:
        raise ValueError(
            f"Owner mapping must contain columns 'brand' and '{owner_col}'. Found: {list(df.columns)}"
        )
    brand_col = cols["brand"]
    owner_col = cols[owner_key]

    df = df[[brand_col, owner_col]].dropna()
    df["brand_norm"] = df[brand_col].map(_norm)
    df["owner_norm"] = df[owner_col].map(_norm)

    dup = df["brand_norm"].duplicated(keep=False)
    if dup.any():
        dups = df.loc[dup, "brand_norm"].unique().tolist()
        raise ValueError(f"Duplicate brand mappings found: {dups[:10]}")

    return dict(zip(df["brand_norm"], df["owner_norm"]))


def load_pricer_map(path: str | Path, *, sheet: str = "brands") -> dict[str, str]:
    return load_owner_map(path, sheet=sheet, owner_col="pricer")


def add_owner_ids(
    product_data: pd.DataFrame,
    owner_map: dict[str, str],
    *,
    firm_col: str = "firm_ids",
    owner_col: str = "owner_ids",
    allow_unmapped: bool = False,
) -> pd.DataFrame:
    df = product_data.copy()
    firm_norm = df[firm_col].astype(str).map(_norm)
    owner = firm_norm.map(owner_map)
    if not allow_unmapped and owner.isna().any():
        missing = sorted(set(firm_norm[owner.isna()].unique()))
        raise ValueError(
            "Unmapped firm_ids in owner map. Sample: " + ", ".join(missing[:10])
        )
    if allow_unmapped:
        owner = owner.fillna(firm_norm)
    df[owner_col] = owner
    return df


def build_owner_ownership(
    aligned_product_data: pd.DataFrame,
    *,
    owner_col: str = "owner_ids",
    market_col: str = "market_ids",
) -> np.ndarray:
    if owner_col not in aligned_product_data.columns:
        raise ValueError(f"{owner_col} not found in aligned_product_data.")
    pd_own = aligned_product_data[[market_col, owner_col]].copy()
    pd_own = pd_own.rename(columns={owner_col: "firm_ids"})
    return np.asarray(pyblp.build_ownership(pd_own), dtype=float)


def attach_ownership_columns(
    aligned_product_data: pd.DataFrame,
    ownership: np.ndarray,
    *,
    prefix: str = "ownership",
) -> pd.DataFrame:
    df = aligned_product_data.copy()
    drop_cols = [c for c in df.columns if c.startswith(prefix)]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    for j in range(ownership.shape[1]):
        df[f"{prefix}{j}"] = ownership[:, j]
    return df
