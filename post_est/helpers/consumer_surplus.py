"""
consumer_surplus.py

Manual consumer surplus (CS) integration over agents for pyblp results, designed to
handle nonlinear demographic interactions in the price coefficient (e.g., price×log(income)).

This implements the standard random-coefficients logit (inclusive value / log-sum) welfare:
  CS_m(gamma) = E_i[ ( log(exp(gamma) + sum_j exp(V_ij)) - gamma ) / (-alpha_i) ].

Key implementation trick for counterfactual prices:
- We do NOT rely on pyblp's compute_consumer_surpluses.
- We hold xi fixed (via baseline delta) and update utilities using:
    delta_cf = delta0 + beta_price * (p_cf - p0)
    mu_cf    = mu0    + tastes_price_i * (p_cf - p0)
  where tastes_price_i is the agent-specific deviation in the price coefficient coming from
  Sigma*nodes + Pi*demographics (for the X2 "prices" dimension).

Assumptions (lean, based on typical pyblp internals):
- results.delta is aligned with results.problem.products.* arrays.
- results.problem.products has: market_ids, product_ids, X2
- results.problem.agents has:  market_ids, nodes, demographics, weights (or weights omitted)
- The mean price coefficient (beta_price) is in X1. You must provide it, or provide its index in beta.
- The price random coefficient dimension exists in X2. You must provide price_x2_index.

Inputs expected from your CF pipeline:
- product_table: DataFrame with columns [market_ids, product_ids, p0, p_cf] at minimum.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

def _results_product_ids(prods, *, preferred: str = "product_ids") -> np.ndarray:
    """Return 1D string ids for products inside a pyblp Products object.

    In some pyblp setups, `product_ids` may be empty; many pipelines store IDs in
    `clustering_ids`. We prefer `preferred` but fall back to `clustering_ids`.
    """
    if hasattr(prods, preferred):
        a = np.asarray(getattr(prods, preferred)).astype(str).ravel()
        if a.size:
            return a
    if hasattr(prods, "clustering_ids"):
        a = np.asarray(getattr(prods, "clustering_ids")).astype(str).ravel()
        if a.size:
            return a
    raise KeyError("No usable product id vector found in results.problem.products (tried preferred and clustering_ids).")


# ----------------------------
# Small numeric helpers
# ----------------------------

def _logsumexp_with_outside(V: np.ndarray, gamma: float = 0.0) -> np.ndarray:
    """
    Compute log(exp(gamma) + sum_j exp(V_ij)) for each i, stably.
    V: (I, J)
    Returns: (I,)
    """
    # max over inside and outside utility
    m = np.maximum(gamma, np.max(V, axis=1))
    # exp(gamma - m) + sum exp(V - m)
    denom = np.exp(gamma - m) + np.exp(V - m[:, None]).sum(axis=1)
    return m + np.log(denom)


def _weighted_mean(x: np.ndarray, w: np.ndarray | None) -> float:
    if w is None:
        return float(np.mean(x))
    wsum = float(np.sum(w))
    return float(np.sum(w * x) / wsum) if wsum > 0 else float("nan")

def _income_levels(income: np.ndarray, transform) -> np.ndarray:
    if transform is None:
        return income
    if callable(transform):
        return np.asarray(transform(income), dtype=float)
    if transform == "log_10k":
        return np.exp(income) * 10_000.0
    raise ValueError(f"Unknown income_transform: {transform}")


# ----------------------------
# Core pieces: tastes & mapping
# ----------------------------

def agent_tastes_for_market(
    results,
    market_id,
    *,
    sigma: np.ndarray | None = None,
    pi: np.ndarray | None = None,
):
    """
    Return agent tastes for X2 dimensions in a given market:
      tastes = nodes @ sigma.T + demographics @ pi.T   (I, K2)

    Always returns:
      tastes : (I, K2) ndarray  (even if sigma/pi are missing -> zeros)
      weights: (I,) ndarray or None

    Notes
    -----
    pyblp market_ids are often strings; we match by string form.
    """
    agents = results.problem.agents

    a_mid = np.asarray(getattr(agents, "market_ids")).astype(str).ravel()
    a_mask = (a_mid == str(market_id))

    nodes_all = np.asarray(agents.nodes, dtype=float)
    nodes = nodes_all[a_mask, :]           # (I, K2)
    demos_all = np.asarray(agents.demographics, dtype=float)
    demos = demos_all[a_mask, :]    # (I, D)
    I = int(nodes.shape[0])
    K2 = int(nodes.shape[1]) if nodes.ndim == 2 else 0

    if sigma is None:
        sigma = getattr(results, "sigma", None)
    if pi is None:
        pi = getattr(results, "pi", None)

    # If sigma/pi are unavailable, treat that component as zero.
    if sigma is None:
        taste_sigma = np.zeros((I, K2), dtype=float)
    else:
        taste_sigma = nodes @ np.asarray(sigma, dtype=float).T

    if pi is None:
        taste_pi = np.zeros((I, K2), dtype=float)
    else:
        taste_pi = demos @ np.asarray(pi, dtype=float).T

    tastes = taste_sigma + taste_pi

    weights = getattr(agents, "weights", None)
    if weights is not None:
        weights = np.asarray(weights, dtype=float).ravel()[a_mask]

    return tastes, weights


def market_products_for_results(
    results,
    market_id,
    *,
    market_col="market_ids",
    id_col="product_ids",
    results_id_col: str = "product_ids",
):
    """
    Pull product ids, delta0, and X2 for a given market in *results' internal order*.
    """
    prods = results.problem.products
    p_mid = np.asarray(getattr(prods, market_col)).astype(str).ravel()
    p_mask = (p_mid == str(market_id))

    ids_all = _results_product_ids(prods, preferred=results_id_col)
    ids = ids_all[p_mask]  # (J,)
    delta0 = np.asarray(results.delta, dtype=float).reshape(-1)[p_mask]  # (J,)
    X2 = np.asarray(prods.X2, dtype=float)[p_mask, :]  # (J, K2)

    return ids, delta0, X2


def align_prices_to_results_order(
    product_table: pd.DataFrame,
    market_id,
    ids_in_results_order: np.ndarray,
    *,
    market_col="market_ids",
    id_col="product_ids",
    p0_col="p0",
    pcf_col="p_cf",
):
    """
    Given a product_table and the ids in results' internal order, return p0 and p_cf
    aligned to that order.
    """
    cols = [id_col, p0_col, pcf_col]
    if "p0_net" in product_table.columns and "p_cf_net" in product_table.columns:
        cols += ["p0_net", "p_cf_net"]
    sub = product_table.loc[product_table[market_col] == market_id, cols].copy()
    sub[id_col] = sub[id_col].astype(str)

    p0_map = sub.set_index(id_col)[p0_col]
    pcf_map = sub.set_index(id_col)[pcf_col]
    if "p0_net" in sub.columns and "p_cf_net" in sub.columns:
        p0_map = sub.set_index(id_col)["p0_net"]
        pcf_map = sub.set_index(id_col)["p_cf_net"]

    p0 = pd.to_numeric(p0_map.reindex(ids_in_results_order), errors="coerce").to_numpy(dtype=float)
    pcf = pd.to_numeric(pcf_map.reindex(ids_in_results_order), errors="coerce").to_numpy(dtype=float)

    return p0, pcf


# ----------------------------
# Main CS calculators
# ----------------------------

def market_cs_manual(
    results,
    product_table: pd.DataFrame,
    market_id,
    *,
    price_x2_index: int,
    beta_price: float | None = None,
    beta_price_index: int | None = None,
    gamma: float = 0.0,
    normalize_weights: bool = True,
    drop_nonnegative_alpha: bool = True,
    alpha_abs_min: float = 1e-6,
    market_col="market_ids",
    id_col="product_ids",
    p0_col="p0",
    pcf_col="p_cf",
):
    """
    Compute CS0, CS_cf, and dCS for one market by integrating over agents.

    Parameters
    ----------
    price_x2_index : column index in X2 corresponding to (raw) price
    beta_price     : mean price coefficient (scalar). If None, uses beta_price_index into results.beta.
    beta_price_index : index into results.beta if beta_price not provided.
    gamma          : outside good utility level (0.0 corresponds to standard normalization)
    drop_nonnegative_alpha : drop agents with alpha_i >= 0 (or |alpha_i| too small)
    alpha_abs_min  : drop if |alpha_i| < alpha_abs_min

    Returns
    -------
    dict with keys:
      CS0, CS_cf, dCS (all per-capita, in "price units" e.g., USD/100k if prices are scaled)
      diag: diagnostics (counts and alpha stats)
    """
    # --- mean price coefficient ---
    if beta_price is None:
        if beta_price_index is None:
            raise ValueError("Provide beta_price or beta_price_index to locate it in results.beta.")
        beta_price = float(np.asarray(results.beta).reshape(-1)[beta_price_index])

    # --- products in results order ---
    ids, delta0, X2 = market_products_for_results(results, market_id, market_col=market_col, id_col=id_col, results_id_col='clustering_ids')
    p0, pcf = align_prices_to_results_order(
        product_table, market_id, ids, market_col=market_col, id_col=id_col, p0_col=p0_col, pcf_col=pcf_col
    )
    dp = pcf - p0

    # --- agent tastes and weights ---
    tastes, w = agent_tastes_for_market(results, market_id)
    # tastes: (I, K2). price deviation for each agent:
    taste_price = tastes[:, price_x2_index].reshape(-1)  # (I,)
    alpha = beta_price + taste_price                      # total price coefficient (I,)

    # --- build mu for baseline: (I, J) = tastes @ X2.T
    mu0 = tastes @ X2.T

    # --- counterfactual updates without rebuilding X2:
    # mean part: delta_cf = delta0 + beta_price * dp
    delta_cf = delta0 + beta_price * dp
    # heterogeneous part: add taste_price_i * dp_j
    mu_cf = mu0 + np.outer(taste_price, dp)

    # --- deterministic utilities ---
    V0 = delta0[None, :] + mu0
    Vcf = delta_cf[None, :] + mu_cf

    # --- inclusive values ---
    logsum0 = _logsumexp_with_outside(V0, gamma=gamma) - gamma
    logsumcf = _logsumexp_with_outside(Vcf, gamma=gamma) - gamma

    # --- filter problematic alpha values ---
    keep = np.ones_like(alpha, dtype=bool)
    if drop_nonnegative_alpha:
        keep &= (alpha < 0)
    keep &= (np.abs(alpha) >= alpha_abs_min)

    if not np.all(keep):
        logsum0 = logsum0[keep]
        logsumcf = logsumcf[keep]
        alpha_k = alpha[keep]
        w_k = None if w is None else w[keep]
    else:
        alpha_k = alpha
        w_k = w

    # CS_i = logsum / (-alpha_i)
    cs0_i = logsum0 / (-alpha_k)
    cscf_i = logsumcf / (-alpha_k)

    # integrate
    if w_k is None:
        CS0 = float(np.mean(cs0_i))
        CS_cf = float(np.mean(cscf_i))
    else:
        if normalize_weights:
            w_use = w_k / np.sum(w_k)
        else:
            w_use = w_k
        CS0 = float(np.sum(w_use * cs0_i))
        CS_cf = float(np.sum(w_use * cscf_i))

    dCS = CS_cf - CS0

    diag = {
        "n_agents_total": int(len(alpha)),
        "n_agents_used": int(len(alpha_k)),
        "share_agents_dropped": float(1.0 - len(alpha_k) / max(1, len(alpha))),
        "alpha_mean_used": float(np.mean(alpha_k)),
        "alpha_min_used": float(np.min(alpha_k)),
        "alpha_max_used": float(np.max(alpha_k)),
    }

    return {"CS0": CS0, "CS_cf": CS_cf, "dCS": dCS, "diag": diag}


def cs_manual_all_markets(
    results,
    product_table: pd.DataFrame,
    *,
    price_x2_index: int,
    beta_price: float | None = None,
    beta_price_index: int | None = None,
    gamma: float = 0.0,
    market_col="market_ids",
    id_col="product_ids",
    p0_col="p0",
    pcf_col="p_cf",
):
    """
    Compute manual CS for every market present in product_table.

    Returns
    -------
    market_cs_df : DataFrame with [market_id, CS0, CS_cf, dCS] + diagnostics cols
    """
    markets = pd.unique(product_table[market_col])

    rows = []
    for mid in markets:
        out = market_cs_manual(
            results, product_table, mid,
            price_x2_index=price_x2_index,
            beta_price=beta_price,
            beta_price_index=beta_price_index,
            gamma=gamma,
            market_col=market_col,
            id_col=id_col,
            p0_col=p0_col,
            pcf_col=pcf_col,
        )
        diag = out.pop("diag")
        rows.append({market_col: mid, **out, **{f"diag_{k}": v for k, v in diag.items()}})

    return pd.DataFrame(rows)


def attach_manual_cs_to_cf_output(
    cf_out: dict,
    results,
    *,
    price_x2_index: int,
    beta_price: float | None = None,
    beta_price_index: int | None = None,
    gamma: float = 0.0,
    market_col="market_ids",
):
    """
    Convenience wrapper for your pipeline output from run_cf_and_summarize.

    Parameters
    ----------
    cf_out : dict returned by run_cf_and_summarize (must include 'product_table')
    results: pyblp ProblemResults
    price_x2_index / beta_price: as in market_cs_manual

    Returns
    -------
    updated_cf_out : copy of cf_out with:
      - 'market_surplus_table_manual' (DataFrame)
      - 'overall_surplus_manual' (1-row DataFrame with totals in per-capita units)
    """
    product_table = cf_out["product_table"]
    mcs = cs_manual_all_markets(
        results, product_table,
        price_x2_index=price_x2_index,
        beta_price=beta_price,
        beta_price_index=beta_price_index,
        gamma=gamma,
        market_col=market_col,
    )

    overall = pd.DataFrame([{
        "CS0_total_percap": float(mcs["CS0"].sum()),
        "CS_cf_total_percap": float(mcs["CS_cf"].sum()),
        "dCS_total_percap": float(mcs["dCS"].sum()),
        "gamma": float(gamma),
    }])

    out = dict(cf_out)
    out["market_surplus_table_manual"] = mcs
    out["overall_surplus_manual"] = overall
    return out

# ---------- helpers ----------
def weighted_quantile(x, q, w=None):
    """Weighted quantiles of x at q in [0,1]."""
    x = np.asarray(x, float)
    q = np.asarray(q, float)
    if w is None:
        return np.quantile(x, q)
    w = np.asarray(w, float)
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x, w = x[m], w[m]
    if x.size == 0:
        return np.full_like(q, np.nan, dtype=float)
    idx = np.argsort(x)
    x, w = x[idx], w[idx]
    cw = np.cumsum(w)
    cw = cw / cw[-1]
    return np.interp(q, cw, x)

def logsumexp_with_outside(V, gamma=0.0):
    """
    V: (I,J). returns inclusive value per agent: log(exp(gamma)+sum_j exp(V_ij)) - gamma
    """
    # stabilize
    vmax = np.max(np.concatenate([V, np.full((V.shape[0], 1), gamma)], axis=1), axis=1, keepdims=True)
    exp_out = np.exp(gamma - vmax[:, 0])
    exp_in  = np.exp(V - vmax)
    s = exp_out + np.sum(exp_in, axis=1)
    return (np.log(s) + vmax[:, 0] - gamma)

# ---------- main function ----------
def cs_change_by_income_bins(
    results,
    product_table,                # out["product_table"]
    market_id,
    *,
    # YOU set these:
    price_x2_index,               # which X2 column is price
    beta_price_index,             # which beta element is mean price coefficient
    income_demo_index=0,          # which demographics column is income (or income-like)
    income_transform=None,        # None or "log_10k" (log income in 10k units -> dollars)
    n_bins=5,                     # quintiles by default
    gamma=0.0,                    # outside good "gamma" if you use it
    drop_nonnegative_alpha=True,  # recommended
    market_col="market_ids",
    id_col="product_ids",
    p0_col="p0",
    pcf_col="p_cf",
    results_market_col="market_ids",
    results_id_col="clustering_ids",  # IMPORTANT: your results IDs live here
):
    # --- pull agents in this market ---
    agents = results.problem.agents
    a_mid = np.asarray(getattr(agents, results_market_col)).astype(str).ravel()
    a_mask = (a_mid == str(market_id))

    nodes = np.asarray(agents.nodes, float)[a_mask, :]           # (I,K2)
    demos = np.asarray(agents.demographics, float)[a_mask, :]    # (I,D)
    w = getattr(agents, "weights", None)
    w = None if w is None else np.asarray(w, float).ravel()[a_mask]
    if w is not None and w.sum() > 0:
        w = w / w.sum()

    # agent tastes for X2
    sigma = np.asarray(results.sigma, float)
    pi = np.asarray(results.pi, float)
    tastes = nodes @ sigma.T + demos @ pi.T                      # (I,K2)

    # --- pull products in this market in RESULTS order ---
    prods = results.problem.products
    p_mid = np.asarray(getattr(prods, results_market_col)).astype(str).ravel()
    p_mask = (p_mid == str(market_id))

    # IDs from results: use clustering_ids (your setup)
    ids_all = np.asarray(getattr(prods, results_id_col)).astype(str).ravel()
    ids = ids_all[p_mask]                                        # (J,)

    X2 = np.asarray(prods.X2, float)[p_mask, :]                  # (J,K2)
    delta0 = np.asarray(results.delta, float).reshape(-1)[p_mask]# (J,)

    # --- align p0 / p_cf from product_table to RESULTS order ---
    cols = [id_col, p0_col, pcf_col]
    if "p0_net" in product_table.columns and "p_cf_net" in product_table.columns:
        cols += ["p0_net", "p_cf_net"]
    sub = product_table.loc[product_table[market_col].astype(str) == str(market_id),
                            cols].copy()
    sub[id_col] = sub[id_col].astype(str)
    sub = sub.set_index(id_col)

    p0  = sub.reindex(ids)[p0_col].to_numpy(float)
    pcf = sub.reindex(ids)[pcf_col].to_numpy(float)
    if "p0_net" in sub.columns and "p_cf_net" in sub.columns:
        p0 = sub.reindex(ids)["p0_net"].to_numpy(float)
        pcf = sub.reindex(ids)["p_cf_net"].to_numpy(float)

    if np.isnan(p0).any() or np.isnan(pcf).any():
        raise ValueError("Price alignment produced NaNs. Check that product_table.product_ids matches results.clustering_ids.")

    dp = pcf - p0

    # --- mean beta price ---
    beta = np.asarray(results.beta, float).reshape(-1)
    beta_price = float(beta[beta_price_index])

    # --- utilities ---
    mu0 = tastes @ X2.T                                           # (I,J)
    tastes_price = tastes[:, price_x2_index]                       # (I,)
    mu_cf = mu0 + np.outer(tastes_price, dp)                       # update only price dim
    delta_cf = delta0 + beta_price * dp                            # update mean price effect

    V0  = delta0[None, :] + mu0
    Vcf = delta_cf[None, :] + mu_cf

    iv0  = logsumexp_with_outside(V0,  gamma=gamma)                # (I,)
    ivcf = logsumexp_with_outside(Vcf, gamma=gamma)                # (I,)

    alpha_i = beta_price + tastes_price                            # (I,)

    # drop bad alphas if requested
    keep = np.isfinite(alpha_i) & np.isfinite(iv0) & np.isfinite(ivcf)
    if drop_nonnegative_alpha:
        keep &= (alpha_i < 0)

    iv0, ivcf, alpha_i = iv0[keep], ivcf[keep], alpha_i[keep]
    incomes_raw = demos[keep, income_demo_index]
    incomes = _income_levels(incomes_raw, income_transform)
    ww = None if w is None else w[keep]
    if ww is not None and ww.sum() > 0:
        ww = ww / ww.sum()

    cs0_i  = iv0  / (-alpha_i)
    cscf_i = ivcf / (-alpha_i)

    # --- weighted income bins ---
    cuts = weighted_quantile(incomes, np.linspace(0, 1, n_bins + 1), w=ww)
    # make strictly increasing (avoid edge issues if many ties)
    cuts[0]  = -np.inf
    cuts[-1] =  np.inf
    bin_id = np.clip(np.searchsorted(cuts[1:-1], incomes, side="right"), 0, n_bins-1)

    rows = []
    for b in range(n_bins):
        m = (bin_id == b)
        if not np.any(m):
            continue
        wb = None if ww is None else ww[m]
        def wmean(x):
            if wb is None:
                return float(np.mean(x[m]))
            return float(np.sum(x[m] * wb) / np.sum(wb))
        CS0  = wmean(cs0_i)
        CScf = wmean(cscf_i)
        dCS  = CScf - CS0
        pct  = np.nan if CS0 == 0 else 100.0 * dCS / CS0
        rows.append({
            "bin": b+1,
            "CS0": CS0,
            "CS_cf": CScf,
            "dCS": dCS,
            "pct_change_vs_baseline": pct,
            "income_min": float(np.nanmin(incomes[m])),
            "income_max": float(np.nanmax(incomes[m])),
            "weight_mass": float(np.sum(wb) if wb is not None else np.mean(m))
        })

    return pd.DataFrame(rows)

def cs_change_by_region_income_bins(
    results,
    product_table,
    market_id,
    *,
    price_x2_index,
    beta_price_index,
    income_demo_index=0,
    income_transform=None,
    region_start_index=1,
    region_labels=None,
    n_bins=5,
    gamma=0.0,
    drop_nonnegative_alpha=True,
    market_col="market_ids",
    id_col="product_ids",
    p0_col="p0",
    pcf_col="p_cf",
    results_market_col="market_ids",
    results_id_col="clustering_ids",
):
    """
    Compute CS changes by region and income bin.

    Expects region indicators in demographics starting at region_start_index,
    with one column per region. Income bins are computed within each region.
    Adds an aggregate row per region (bin="All").
    """
    agents = results.problem.agents
    a_mid = np.asarray(getattr(agents, results_market_col)).astype(str).ravel()
    a_mask = (a_mid == str(market_id))

    nodes = np.asarray(agents.nodes, float)[a_mask, :]
    demos = np.asarray(agents.demographics, float)[a_mask, :]
    w = getattr(agents, "weights", None)
    w = None if w is None else np.asarray(w, float).ravel()[a_mask]
    if w is not None and w.sum() > 0:
        w = w / w.sum()

    sigma = np.asarray(results.sigma, float)
    pi = np.asarray(results.pi, float)
    tastes = nodes @ sigma.T + demos @ pi.T

    prods = results.problem.products
    p_mid = np.asarray(getattr(prods, results_market_col)).astype(str).ravel()
    p_mask = (p_mid == str(market_id))

    ids_all = np.asarray(getattr(prods, results_id_col)).astype(str).ravel()
    ids = ids_all[p_mask]

    X2 = np.asarray(prods.X2, float)[p_mask, :]
    delta0 = np.asarray(results.delta, float).reshape(-1)[p_mask]

    cols = [id_col, p0_col, pcf_col]
    if "p0_net" in product_table.columns and "p_cf_net" in product_table.columns:
        cols += ["p0_net", "p_cf_net"]
    sub = product_table.loc[
        product_table[market_col].astype(str) == str(market_id),
        cols
    ].copy()
    sub[id_col] = sub[id_col].astype(str)
    sub = sub.set_index(id_col)

    p0 = sub.reindex(ids)[p0_col].to_numpy(float)
    pcf = sub.reindex(ids)[pcf_col].to_numpy(float)
    if "p0_net" in sub.columns and "p_cf_net" in sub.columns:
        p0 = sub.reindex(ids)["p0_net"].to_numpy(float)
        pcf = sub.reindex(ids)["p_cf_net"].to_numpy(float)
    if np.isnan(p0).any() or np.isnan(pcf).any():
        raise ValueError("Price alignment produced NaNs. Check product_table.product_ids vs results ids.")

    dp = pcf - p0

    beta = np.asarray(results.beta, float).reshape(-1)
    beta_price = float(beta[beta_price_index])

    mu0 = tastes @ X2.T
    tastes_price = tastes[:, price_x2_index]
    mu_cf = mu0 + np.outer(tastes_price, dp)
    delta_cf = delta0 + beta_price * dp

    V0 = delta0[None, :] + mu0
    Vcf = delta_cf[None, :] + mu_cf

    iv0 = logsumexp_with_outside(V0, gamma=gamma)
    ivcf = logsumexp_with_outside(Vcf, gamma=gamma)

    alpha_i = beta_price + tastes_price

    keep = np.isfinite(alpha_i) & np.isfinite(iv0) & np.isfinite(ivcf)
    if drop_nonnegative_alpha:
        keep &= (alpha_i < 0)

    iv0, ivcf, alpha_i = iv0[keep], ivcf[keep], alpha_i[keep]
    demos = demos[keep, :]
    w = None if w is None else w[keep]
    if w is not None and w.sum() > 0:
        w = w / w.sum()

    cs0_i = iv0 / (-alpha_i)
    cscf_i = ivcf / (-alpha_i)

    incomes_raw = demos[:, income_demo_index]
    incomes = _income_levels(incomes_raw, income_transform)

    if region_labels is None:
        region_labels = [f"Region {i+1}" for i in range(6)]

    n_regions = len(region_labels)
    region_mat = demos[:, region_start_index:region_start_index + n_regions]

    rows = []
    for r_idx, r_label in enumerate(region_labels):
        r_mask = region_mat[:, r_idx] > 0.5
        if not np.any(r_mask):
            continue

        inc_r = incomes[r_mask]
        cs0_r = cs0_i[r_mask]
        cscf_r = cscf_i[r_mask]
        w_r = None if w is None else w[r_mask]
        if w_r is not None and w_r.sum() > 0:
            w_r = w_r / w_r.sum()

        cuts = weighted_quantile(inc_r, np.linspace(0, 1, n_bins + 1), w=w_r)
        cuts[0] = -np.inf
        cuts[-1] = np.inf
        bin_id = np.clip(np.searchsorted(cuts[1:-1], inc_r, side="right"), 0, n_bins - 1)

        # Aggregate for the region (all incomes)
        def wmean_all(x):
            if w_r is None:
                return float(np.mean(x))
            den = float(np.sum(w_r))
            return float(np.sum(x * w_r) / den) if den > 0 else float("nan")

        CS0_all = wmean_all(cs0_r)
        CScf_all = wmean_all(cscf_r)
        dCS_all = CScf_all - CS0_all
        pct_all = np.nan if CS0_all == 0 else 100.0 * dCS_all / CS0_all
        weight_mass_all = float(np.sum(w_r) if w_r is not None else 1.0)

        rows.append({
            "region": r_label,
            "bin": "All",
            "CS0": CS0_all,
            "CS_cf": CScf_all,
            "dCS": dCS_all,
            "pct_change_vs_baseline": pct_all,
            "income_min": float(np.nanmin(inc_r)),
            "income_max": float(np.nanmax(inc_r)),
            "weight_mass": weight_mass_all,
        })

        for b in range(n_bins):
            m = (bin_id == b)
            if not np.any(m):
                continue
            wb = None if w_r is None else w_r[m]

            def wmean(x):
                if wb is None:
                    return float(np.mean(x[m]))
                den = float(np.sum(wb))
                return float(np.sum(x[m] * wb) / den) if den > 0 else float("nan")

            CS0 = wmean(cs0_r)
            CScf = wmean(cscf_r)
            dCS = CScf - CS0
            pct = np.nan if CS0 == 0 else 100.0 * dCS / CS0
            weight_mass = float(np.sum(wb) if wb is not None else np.mean(m))

            rows.append({
                "region": r_label,
                "bin": b + 1,
                "CS0": CS0,
                "CS_cf": CScf,
                "dCS": dCS,
                "pct_change_vs_baseline": pct,
                "income_min": float(np.nanmin(inc_r[m])),
                "income_max": float(np.nanmax(inc_r[m])),
                "weight_mass": weight_mass,
            })

    return pd.DataFrame(rows)


def cs_change_by_state(
    results,
    product_table,
    agent_df,
    market_id,
    *,
    price_x2_index,
    beta_price_index,
    income_col="log_income_10k",
    division_cols=None,
    state_col="state",
    year_col="year",
    nodes_prefix="nodes",
    weight_col="weights",
    gamma=0.0,
    drop_nonnegative_alpha=True,
    market_col="market_ids",
    id_col="product_ids",
    p0_col="p0",
    pcf_col="p_cf",
    results_market_col="market_ids",
    results_id_col="clustering_ids",
):
    """
    Compute CS changes by state (weighted average within each state).

    Uses agent_df for nodes, demographics, and weights. Division-specific tastes
    are captured by including division dummy columns in the demographics matrix.
    """
    if division_cols is None:
        division_cols = ["div1", "div2", "div3", "div4", "div5", "div6"]

    a_mask = agent_df[year_col].astype(str).eq(str(market_id))
    agents = agent_df.loc[a_mask].copy()
    if agents.empty:
        raise ValueError("No agent data for market_id/year.")

    node_cols = [c for c in agents.columns if c.startswith(nodes_prefix)]
    if not node_cols:
        raise ValueError("No nodes_* columns found in agent_df.")

    # Align node columns to sigma dimension (K2)
    sigma = np.asarray(results.sigma, float)
    K2 = sigma.shape[0]
    def _node_key(c):
        suf = c[len(nodes_prefix):]
        try:
            return int(suf)
        except ValueError:
            return suf
    node_cols = sorted(node_cols, key=_node_key)
    if len(node_cols) < K2:
        raise ValueError(f"agent_df has {len(node_cols)} nodes columns but sigma expects {K2}.")
    node_cols = node_cols[:K2]

    nodes = agents[node_cols].to_numpy(dtype=float)
    demos = agents[[income_col] + division_cols].to_numpy(dtype=float)

    w = None
    if weight_col in agents.columns:
        w = agents[weight_col].to_numpy(dtype=float)
        if np.isfinite(w).any() and w.sum() > 0:
            w = w / w.sum()

    pi = np.asarray(results.pi, float)
    tastes = nodes @ sigma.T + demos @ pi.T

    prods = results.problem.products
    p_mid = np.asarray(getattr(prods, results_market_col)).astype(str).ravel()
    p_mask = (p_mid == str(market_id))

    ids_all = np.asarray(getattr(prods, results_id_col)).astype(str).ravel()
    ids = ids_all[p_mask]

    X2 = np.asarray(prods.X2, float)[p_mask, :]
    delta0 = np.asarray(results.delta, float).reshape(-1)[p_mask]

    cols = [id_col, p0_col, pcf_col]
    if "p0_net" in product_table.columns and "p_cf_net" in product_table.columns:
        cols += ["p0_net", "p_cf_net"]
    sub = product_table.loc[
        product_table[market_col].astype(str) == str(market_id),
        cols
    ].copy()
    sub[id_col] = sub[id_col].astype(str)
    sub = sub.set_index(id_col)

    p0 = sub.reindex(ids)[p0_col].to_numpy(float)
    pcf = sub.reindex(ids)[pcf_col].to_numpy(float)
    if "p0_net" in sub.columns and "p_cf_net" in sub.columns:
        p0 = sub.reindex(ids)["p0_net"].to_numpy(float)
        pcf = sub.reindex(ids)["p_cf_net"].to_numpy(float)
    if np.isnan(p0).any() or np.isnan(pcf).any():
        raise ValueError("Price alignment produced NaNs. Check product_table.product_ids vs results ids.")

    dp = pcf - p0

    beta = np.asarray(results.beta, float).reshape(-1)
    beta_price = float(beta[beta_price_index])

    mu0 = tastes @ X2.T
    tastes_price = tastes[:, price_x2_index]
    mu_cf = mu0 + np.outer(tastes_price, dp)
    delta_cf = delta0 + beta_price * dp

    V0 = delta0[None, :] + mu0
    Vcf = delta_cf[None, :] + mu_cf

    iv0 = logsumexp_with_outside(V0, gamma=gamma)
    ivcf = logsumexp_with_outside(Vcf, gamma=gamma)

    alpha_i = beta_price + tastes_price

    keep = np.isfinite(alpha_i) & np.isfinite(iv0) & np.isfinite(ivcf)
    if drop_nonnegative_alpha:
        keep &= (alpha_i < 0)

    iv0, ivcf, alpha_i = iv0[keep], ivcf[keep], alpha_i[keep]
    states = agents[state_col].to_numpy(dtype=str)[keep]
    w_use = None if w is None else w[keep]

    cs0_i = iv0 / (-alpha_i)
    cscf_i = ivcf / (-alpha_i)

    rows = []
    for st in pd.unique(states):
        m = states == st
        if not np.any(m):
            continue
        w_st = None if w_use is None else w_use[m]
        if w_st is not None and w_st.sum() > 0:
            w_st = w_st / w_st.sum()

        def wmean(x):
            if w_st is None:
                return float(np.mean(x[m]))
            return float(np.sum(x[m] * w_st))

        CS0 = wmean(cs0_i)
        CScf = wmean(cscf_i)
        dCS = CScf - CS0
        pct = np.nan if CS0 == 0 else 100.0 * dCS / CS0
        rows.append({
            "state": st,
            "CS0": CS0,
            "CS_cf": CScf,
            "dCS": dCS,
            "pct_change_vs_baseline": pct,
            "weight_mass": float(np.sum(w_st) if w_st is not None else np.mean(m)),
        })

    return pd.DataFrame(rows)

import numpy as np
import pandas as pd

def _weighted_quantile(x, q, w=None):
    x = np.asarray(x, float)
    q = np.asarray(q, float)
    if w is None:
        return np.quantile(x, q)
    w = np.asarray(w, float)
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x, w = x[m], w[m]
    idx = np.argsort(x)
    x, w = x[idx], w[idx]
    cw = np.cumsum(w)
    cw = cw / cw[-1]
    return np.interp(q, cw, x)

def _p_out_from_V(V, gamma=0.0):
    """
    V: (I,J) inside utilities. Returns P_out: (I,)
    """
    # stable log-sum-exp for inside
    vmax = V.max(axis=1, keepdims=True)
    inside_sum = np.sum(np.exp(V - vmax), axis=1) * np.exp(vmax[:, 0])
    eg = np.exp(gamma)
    return eg / (eg + inside_sum)

def switch_to_outside_by_income(
    results,
    product_table,
    *,
    market_id,
    price_x2_index,
    beta_price_index,
    income_demo_index=0,
    income_transform=None,
    n_bins=5,
    gamma=0.0,
    # IMPORTANT: results uses clustering_ids; table uses product_ids
    results_market_col="market_ids",
    results_id_col="clustering_ids",
    table_market_col="market_ids",
    table_id_col="product_ids",
    p0_col="p0",
    pcf_col="p_cf",
    # simulation option
    simulate_switches=True,
    seed=0,
):
    """
    Returns a table by income bin with:
      - outside share baseline/cf (expected probs)
      - delta outside share (pp)
      - (optional) simulated % switching inside->outside using common random numbers
    """

    # ---------- agents in market ----------
    agents = results.problem.agents
    a_mid = np.asarray(getattr(agents, results_market_col)).astype(str).ravel()
    a_mask = (a_mid == str(market_id))

    nodes = np.asarray(agents.nodes, float)[a_mask, :]          # (I,K2)
    demos = np.asarray(agents.demographics, float)[a_mask, :]   # (I,D)
    incomes_raw = demos[:, income_demo_index]
    incomes = _income_levels(incomes_raw, income_transform)

    w = getattr(agents, "weights", None)
    w = None if w is None else np.asarray(w, float).ravel()[a_mask]
    if w is not None and w.sum() > 0:
        w = w / w.sum()

    # agent tastes for X2: nodes @ sigma.T + demos @ pi.T
    sigma = np.asarray(results.sigma, float)
    pi    = np.asarray(results.pi, float)
    tastes = nodes @ sigma.T + demos @ pi.T                     # (I,K2)

    # ---------- products in market (results order) ----------
    prods = results.problem.products
    p_mid = np.asarray(getattr(prods, results_market_col)).astype(str).ravel()
    p_mask = (p_mid == str(market_id))

    ids_all = np.asarray(getattr(prods, results_id_col)).astype(str).ravel()
    ids = ids_all[p_mask]                                       # (J,)
    X2 = np.asarray(prods.X2, float)[p_mask, :]                 # (J,K2)
    delta0 = np.asarray(results.delta, float).reshape(-1)[p_mask]  # (J,)

    # prices from product_table aligned to results ids
    cols = [table_id_col, p0_col, pcf_col]
    if "p0_net" in product_table.columns and "p_cf_net" in product_table.columns:
        cols += ["p0_net", "p_cf_net"]
    sub = product_table.loc[
        product_table[table_market_col].astype(str) == str(market_id),
        cols
    ].copy()
    sub[table_id_col] = sub[table_id_col].astype(str)
    sub = sub.set_index(table_id_col)

    p0  = sub.reindex(ids)[p0_col].to_numpy(float)
    pcf = sub.reindex(ids)[pcf_col].to_numpy(float)
    if "p0_net" in sub.columns and "p_cf_net" in sub.columns:
        p0 = sub.reindex(ids)["p0_net"].to_numpy(float)
        pcf = sub.reindex(ids)["p_cf_net"].to_numpy(float)
    if np.isnan(p0).any() or np.isnan(pcf).any():
        raise ValueError("Price alignment produced NaNs. Check product_ids vs clustering_ids mapping.")

    dp = pcf - p0

    beta = np.asarray(results.beta, float).reshape(-1)
    beta_price = float(beta[beta_price_index])

    # ---------- baseline & cf inside utilities ----------
    mu0 = tastes @ X2.T                              # (I,J)
    tastes_price = tastes[:, price_x2_index]         # (I,)
    mu_cf = mu0 + np.outer(tastes_price, dp)         # only price changes

    delta_cf = delta0 + beta_price * dp              # mean price effect

    V0  = delta0[None, :]   + mu0
    Vcf = delta_cf[None, :] + mu_cf

    # ---------- expected outside probabilities ----------
    Pout0  = _p_out_from_V(V0,  gamma=gamma)
    Poutcf = _p_out_from_V(Vcf, gamma=gamma)

    # ---------- income bins (weighted quantiles) ----------
    cuts = _weighted_quantile(incomes, np.linspace(0, 1, n_bins + 1), w=w)
    cuts[0], cuts[-1] = -np.inf, np.inf
    bin_id = np.searchsorted(cuts[1:-1], incomes, side="right")  # 0..n_bins-1

    def wmean(x, mask):
        if w is None:
            return float(np.mean(x[mask]))
        ww = w[mask]
        return float(np.sum(x[mask] * ww) / np.sum(ww))

    rows = []
    for b in range(n_bins):
        m = (bin_id == b)
        if not np.any(m):
            continue
        rows.append({
            "bin": b + 1,
            "outside_share0":  wmean(Pout0,  m),
            "outside_share_cf":wmean(Poutcf,m),
            "delta_outside_pp":100 * (wmean(Poutcf, m) - wmean(Pout0, m)),
            "income_min": float(np.min(incomes[m])),
            "income_max": float(np.max(incomes[m])),
        })

    out_tbl = pd.DataFrame(rows)

    # ---------- optional: simulated discrete switching (inside->outside) ----------
    if simulate_switches:
        rng = np.random.default_rng(seed)

        # EV1/Gumbel shocks for inside goods and outside good; reuse across scenarios
        eps_in  = rng.gumbel(loc=0.0, scale=1.0, size=V0.shape)     # (I,J)
        eps_out = rng.gumbel(loc=0.0, scale=1.0, size=(V0.shape[0],))  # (I,)

        # total utility with shocks
        U0_in  = V0  + eps_in
        Ucf_in = Vcf + eps_in
        U0_out  = gamma + eps_out
        Ucf_out = gamma + eps_out

        # chosen option: outside if U_out >= max inside
        inside0 = np.max(U0_in, axis=1)
        insidecf= np.max(Ucf_in, axis=1)

        choose_out0  = (U0_out  >= inside0)
        choose_outcf = (Ucf_out >= insidecf)

        switch_in_to_out = (~choose_out0) & (choose_outcf)

        # summarize by bins
        sim_rows = []
        for b in range(n_bins):
            m = (bin_id == b)
            if not np.any(m):
                continue
            if w is None:
                sim = float(np.mean(switch_in_to_out[m]))
            else:
                ww = w[m]
                sim = float(np.sum(switch_in_to_out[m] * ww) / np.sum(ww))
            sim_rows.append(sim)

        out_tbl["sim_switch_inside_to_outside"] = 100 * np.array(sim_rows)  # percent

    return out_tbl
