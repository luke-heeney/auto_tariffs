"""Alternative agent prep for division-based agent samples."""

import random
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm, qmc


def load_agent_data(
    product_data: pd.DataFrame,
    first_log: int,
    division_map: Dict[str, int],
    use_division: int = 0,
    use_halton: int = 1,
    csv_path: str = "data/agent_incomes_400perdivision_updated.csv",
) -> Tuple[pd.DataFrame, pd.DataFrame, float, float]:
    """
    Load and preprocess agent-level data with columns [year, division, income, weights].
    """

    agent_data = pd.read_csv(csv_path).copy()

    if "year" in agent_data.columns and "market_ids" not in agent_data.columns:
        agent_data.rename(columns={"year": "market_ids"}, inplace=True)

    required_cols = {"market_ids", "division", "income", "weights"}
    missing = required_cols - set(agent_data.columns)
    if missing:
        raise ValueError(f"agent_data missing required columns: {missing}")

    income_dollars = agent_data["income"].astype(float)
    agent_data["income_raw_10k"] = income_dollars / 10000.0
    agent_data["log_income_10k"] = np.log(np.clip(agent_data["income_raw_10k"], 1e-6, None))

    agent_data["income"] = income_dollars / 100000.0
    agent_data["income_raw"] = agent_data["income"]
    income_mean = agent_data["income_raw"].mean()
    income_std = agent_data["income_raw"].std()
    agent_data["income"] = (agent_data["income_raw"] - income_mean) / income_std

    # Align markets with product_data
    prod_markets = set(product_data["market_ids"])
    agent_markets = set(agent_data["market_ids"])
    common_markets = prod_markets & agent_markets

    agent_data = agent_data[agent_data["market_ids"].isin(common_markets)].copy()
    product_data_aligned = product_data[product_data["market_ids"].isin(common_markets)].copy()

    # Normalize weights within each market to sum to 1
    agent_data["weights"] = agent_data["weights"].astype(float)
    weight_sums = agent_data.groupby("market_ids")["weights"].transform("sum")
    if (weight_sums == 0).any():
        raise ValueError("Encountered zero total weight in a market.")
    agent_data["weights"] = agent_data["weights"] / weight_sums

    # Unobserved heterogeneity nodes0..nodes14
    n_agents = agent_data.shape[0]
    n_nodes = 15

    set_seed = 53
    random.seed(set_seed)
    np.random.seed(set_seed)
    rng = np.random.default_rng(set_seed)

    print(f"Agent data seed: {set_seed}")
    print(f"Using Halton draws for nodes: {bool(use_halton)}")

    if use_halton == 1:
        sampler = qmc.Halton(d=n_nodes, scramble=True, seed=set_seed)
        z = norm.ppf(sampler.random(n_agents))
    else:
        z = np.random.normal(size=(n_agents, n_nodes))

    perm = rng.permutation(n_agents)
    z = z[perm]

    if first_log == 1:
        agent_data["nodes0"] = np.exp(z[:, 0])
    else:
        agent_data["nodes0"] = z[:, 0]

    for k in range(1, n_nodes):
        agent_data[f"nodes{k}"] = z[:, k]

    # Optional: division dummies for micro moments
    if use_division == 1:
        agent_data["division"] = agent_data["division"].astype(str).str.strip()
        agent_data["division_id"] = agent_data["division"].map(division_map)

        if agent_data["division_id"].isna().any():
            missing = agent_data.loc[agent_data["division_id"].isna(), "division"].unique()
            raise ValueError(f"Missing division_id mapping for divisions: {missing}")

        for div_id in sorted(set(division_map.values())):
            col_name = f"div_{div_id}"
            agent_data[col_name] = (agent_data["division_id"] == div_id).astype(float)

    return agent_data, product_data_aligned, income_mean, income_std
