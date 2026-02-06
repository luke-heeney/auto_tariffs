"""
Summary-table helpers for counterfactual analysis.

Extracted from run_cf.ipynb Appendix C.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# -------------------------
# small utilities
# -------------------------

def _as_1d(a) -> np.ndarray:
    return np.asarray(a).reshape(-1)


def _market_mask(ids, market_id) -> np.ndarray:
    v = np.asarray(ids)
    if v.ndim == 2 and v.shape[1] == 1:
        v = v[:, 0]
    return v.astype(str) == str(market_id)


def _normalize_weights(w: np.ndarray) -> np.ndarray:
    w = _as_1d(w).astype(float)
    s = w.sum()
    return (w / s) if (np.isfinite(s) and s > 0) else (np.ones_like(w) / len(w))


def weighted_quantile(x: np.ndarray, q: np.ndarray, w: np.ndarray) -> np.ndarray:
    x = _as_1d(x).astype(float)
    w = _normalize_weights(w)
    o = np.argsort(x)
    xs, ws = x[o], w[o]
    cdf = np.cumsum(ws)
    return np.interp(q, cdf, xs)


def _income_levels(income, transform):
    if transform is None:
        return np.asarray(income, dtype=float)
    if callable(transform):
        return np.asarray(transform(income), dtype=float)
    if transform == "log_10k":
        return np.exp(income) * 10_000.0
    raise ValueError(f"Unknown income_transform: {transform}")


# -------------------------
# pull market objects
# -------------------------

def market_agents(results, market_id):
    agents = results.problem.agents
    m = _market_mask(agents.market_ids, market_id)
    nodes = np.asarray(agents.nodes, dtype=float)[m, :]
    demos = np.asarray(agents.demographics, dtype=float)[m, :]
    w = getattr(agents, "weights", None)
    if w is None:
        w = np.ones(nodes.shape[0], dtype=float) / nodes.shape[0]
    else:
        w = np.asarray(w, dtype=float)
        if w.ndim == 2 and w.shape[1] == 1:
            w = w[:, 0]
        w = _normalize_weights(w[m])
    return nodes, demos, w, m


def market_products(results, market_id, *, product_id_field="clustering_ids"):
    prods = results.problem.products
    pm = _market_mask(prods.market_ids, market_id)

    ids_raw = np.asarray(getattr(prods, product_id_field))
    if ids_raw.ndim == 2 and ids_raw.shape[1] == 1:
        ids_raw = ids_raw[:, 0]
    ids = ids_raw.astype(str)[pm]

    delta0 = np.asarray(results.delta, dtype=float)
    if delta0.ndim == 2 and delta0.shape[1] == 1:
        delta0 = delta0[:, 0]
    delta0 = delta0[pm]

    X2 = np.asarray(prods.X2, dtype=float)[pm, :]
    return ids, delta0, X2


def align_prices_from_product_table(
    product_table: pd.DataFrame,
    market_id,
    ids_results: np.ndarray,
    *,
    market_col="market_ids",
    table_id_col="product_ids",
    p0_col="p0",
    pcf_col="p_cf",
):
    dfm = product_table.loc[
        product_table[market_col].astype(str).to_numpy() == str(market_id),
        [table_id_col, p0_col, pcf_col],
    ].copy()
    dfm[table_id_col] = dfm[table_id_col].astype(str)
    dfm = dfm.set_index(table_id_col)

    p0 = dfm.reindex(ids_results)[p0_col].to_numpy(dtype=float)
    pcf = dfm.reindex(ids_results)[pcf_col].to_numpy(dtype=float)
    if np.any(~np.isfinite(p0)) or np.any(~np.isfinite(pcf)):
        raise ValueError("Non-finite prices after alignment. Check id mapping between product_table and results.")
    return p0, pcf


# -------------------------
# tastes + manual CS (Grieco eq. 8)
# -------------------------

def agent_tastes_for_market(results, market_id):
    """
    tastes = nodes @ sigma.T + demos @ pi.T   (I, K2)
    """
    nodes, demos, w, _ = market_agents(results, market_id)
    sigma = np.asarray(results.sigma, dtype=float)
    pi = np.asarray(results.pi, dtype=float)
    tastes = nodes @ sigma.T + demos @ pi.T
    return tastes, w, demos


def cs_by_income_quintile(
    results,
    product_table: pd.DataFrame,
    *,
    market_id,
    price_x2_index: int,
    beta_price_index: int,
    income_demo_index: int,
    income_transform,
    gamma: float,
    total_market_size: float,
    price_scale_usd_per_unit: float = 100_000.0,
    product_id_field_in_results: str = "clustering_ids",
    table_id_col: str = "product_ids",
):
    """
    Returns DataFrame (Q1..Q5) with CS0/CS_cf/dCS in millions USD and % change.
    """
    ids, delta0, X2 = market_products(results, market_id, product_id_field=product_id_field_in_results)
    p0, pcf = align_prices_from_product_table(
        product_table, market_id, ids, table_id_col=table_id_col
    )
    dp = pcf - p0

    tastes, w, demos = agent_tastes_for_market(results, market_id)

    beta = _as_1d(results.beta).astype(float)
    beta_price = float(beta[beta_price_index])

    taste_price = tastes[:, price_x2_index]
    alpha = beta_price + taste_price  # total price coefficient

    # drop weird agents
    keep = np.isfinite(alpha) & (alpha < 0) & (np.abs(alpha) > 1e-8)
    alpha = alpha[keep]
    w = _normalize_weights(w[keep])
    tastes = tastes[keep, :]
    demos = demos[keep, :]

    mu0 = tastes @ X2.T
    mu_cf = mu0 + np.outer(tastes[:, price_x2_index], dp)
    delta_cf = delta0 + beta_price * dp

    def logsum(V):
        # stable log(exp(gamma) + sum exp(V))
        m = np.maximum(gamma, V.max(axis=1))
        return m + np.log(np.exp(gamma - m) + np.sum(np.exp(V - m[:, None]), axis=1))

    v0 = delta0[None, :] + mu0
    vcf = delta_cf[None, :] + mu_cf

    ls0 = logsum(v0)
    lscf = logsum(vcf)

    cs0_i = (-1.0 / alpha) * (ls0 - gamma)
    cscf_i = (-1.0 / alpha) * (lscf - gamma)

    inc_raw = demos[:, income_demo_index].astype(float)
    inc = _income_levels(inc_raw, income_transform)
    cuts = weighted_quantile(inc, np.array([0.2, 0.4, 0.6, 0.8]), w)

    qbin = np.ones(len(inc), dtype=int)
    qbin[inc > cuts[0]] = 2
    qbin[inc > cuts[1]] = 3
    qbin[inc > cuts[2]] = 4
    qbin[inc > cuts[3]] = 5

    factor_musd = float(total_market_size) * (price_scale_usd_per_unit / 1_000_000.0)

    rows = []
    for k in range(1, 6):
        m = (qbin == k)
        cs0_bin = float(np.sum(w[m] * cs0_i[m])) * factor_musd
        cscf_bin = float(np.sum(w[m] * cscf_i[m])) * factor_musd
        dcs = cscf_bin - cs0_bin
        pct = np.nan if cs0_bin == 0 else 100.0 * dcs / cs0_bin
        rows.append({"Income quintile": f"Q{k}", "CS0": cs0_bin, "CS_cf": cscf_bin, "Δ": dcs, "%Δ": pct})
    return pd.DataFrame(rows).set_index("Income quintile")


# -------------------------
# producer-side pieces
# -------------------------

def producer_surplus(out: dict) -> tuple[float, float]:
    ft = out["firm_table"]
    d = float(ft["dpi_millions_usd"].sum())
    base = float(ft["pi0_millions_usd"].sum())
    pct = np.nan if base == 0 else 100.0 * d / base
    return d, pct


def total_cars_sold(out: dict, *, total_market_size: float) -> tuple[float, float]:
    pt = out["product_table"]
    base = float(total_market_size) * float(pt["s0"].sum())
    cf = float(total_market_size) * float(pt["s_cf"].sum())
    d = cf - base
    pct = np.nan if base == 0 else 100.0 * d / base
    return d, pct


def producer_surplus_by_firm_origin(
    out: dict,
    product_data: pd.DataFrame,
    *,
    firm_col="firm_ids",
    origin_col="home_mkt",
):
    """
    Uses product_data[home_mkt] to infer US firm:
      firm is US if mean(home_mkt) > 0.5 across its products.
    """
    firm_is_us = (product_data.groupby(firm_col)[origin_col].mean() > 0.5)

    ft = out["firm_table"].set_index(firm_col)
    common = ft.index.intersection(firm_is_us.index)
    ft = ft.loc[common]
    is_us = firm_is_us.loc[common].astype(bool)

    def agg(mask):
        base = float(ft.loc[mask, "pi0_millions_usd"].sum())
        d = float(ft.loc[mask, "dpi_millions_usd"].sum())
        pct = np.nan if base == 0 else 100.0 * d / base
        return d, pct

    d_us, pct_us = agg(is_us)
    d_non, pct_non = agg(~is_us)
    return {"US firms": (d_us, pct_us), "Non-US firms": (d_non, pct_non)}


# -------------------------
# full table builder
# -------------------------

def build_qje_style_summary_table(
    scenarios: dict[str, dict],
    *,
    results,
    product_data: pd.DataFrame,
    market_id,
    price_x2_index: int,
    beta_price_index: int,
    income_demo_index: int,
    income_transform,
    total_market_size: float,
    price_scale_usd_per_unit: float = 100_000.0,
    gamma: float = 0.0,
    product_id_field_in_results: str = "clustering_ids",
    table_id_col: str = "product_ids",
) -> pd.DataFrame:
    """
    Returns a MultiIndex DataFrame with rows (Panel, Outcome) and columns:
      scenario x {Δ, %Δ}
    """
    rows = []

    # Panel A
    for outcome in ["Producer Surplus", "Total Cars Sold (units)"]:
        row = {"Panel": "Panel A. Aggregate outcomes", "Outcome": outcome}
        for name, out in scenarios.items():
            if outcome == "Producer Surplus":
                d, pct = producer_surplus(out)
            else:
                d, pct = total_cars_sold(out, total_market_size=total_market_size)
            row[(name, "Δ")] = d
            row[(name, "%Δ")] = pct
        rows.append(row)

    # Panel A2 (firm origin)
    for firm_group in ["US firms", "Non-US firms"]:
        row = {"Panel": "Panel A2. Producer surplus by firm origin (millions USD)", "Outcome": firm_group}
        for name, out in scenarios.items():
            mp = producer_surplus_by_firm_origin(out, product_data)
            d, pct = mp[firm_group]
            row[(name, "Δ")] = d
            row[(name, "%Δ")] = pct
        rows.append(row)

    # Panel B (manual CS quintiles)
    for name, out in scenarios.items():
        csq = cs_by_income_quintile(
            results,
            out["product_table"],
            market_id=market_id,
            price_x2_index=price_x2_index,
            beta_price_index=beta_price_index,
            income_demo_index=income_demo_index,
            income_transform=income_transform,
            gamma=gamma,
            total_market_size=total_market_size,
            price_scale_usd_per_unit=price_scale_usd_per_unit,
            product_id_field_in_results=product_id_field_in_results,
            table_id_col=table_id_col,
        )
        scenarios[name]["_cs_quintiles"] = csq  # stash for reuse if you want

    for q in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
        row = {"Panel": "Panel B. Consumer surplus by income quintile (millions USD)", "Outcome": q}
        for name in scenarios:
            csq = scenarios[name]["_cs_quintiles"]
            row[(name, "Δ")] = float(csq.loc[q, "Δ"])
            row[(name, "%Δ")] = float(csq.loc[q, "%Δ"])
        rows.append(row)

    df = pd.DataFrame(rows)

    scenario_names = list(scenarios.keys())

    # Build MultiIndex columns that *already include* Panel/Outcome
    cols = [("", "Panel"), ("", "Outcome")]
    for name in scenario_names:
        cols.extend([(name, "Δ"), (name, "%Δ")])

    df.columns = pd.MultiIndex.from_tuples(cols)

    # Now set the index using the MultiIndex column labels
    df = df.set_index([("", "Panel"), ("", "Outcome")])
    return df


def format_table(df: pd.DataFrame, *, delta_decimals=1, pct_decimals=1) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if c[1] == "Δ":
            out[c] = out[c].map(lambda x: "" if pd.isna(x) else f"{x:+.{delta_decimals}f}")
        else:
            out[c] = out[c].map(lambda x: "" if pd.isna(x) else f"{x:+.{pct_decimals}f}")
    return out
