import numpy as np
import pandas as pd
import pyblp


def build_cex_moments(
    agent_data: pd.DataFrame,
    product_data: pd.DataFrame,
    income_col: int = 0,
    age_col: int = 2,
    N_observed: int = 1000,
    N_pre: int = 500,
    N_cov: int = 500,
    N_pool: int = 1000,
):
    """
    Build CEX-based micro moments:
      - income quintile purchase-probability ratios (2015–2024 pooled)
      - income quintile price differences (pre, covid, pooled)
      - 2015 age × price differences.

    Returns
    -------
    (
      micro_moments_income_quintiles,
      micro_price_diff_pre,
      micro_price_diff_cov,
      micro_price_diff_pool,
      micro_age_price_diff_2015,
      quintile_cutoffs_raw,
    )
    """

    INCOME_COL = income_col

    ##############################
    # CEX Micro moments - INCOME QUINTILES
    ##############################

    valid_markets = set(
        product_data.loc[
            product_data["market_ids"].between(2015, 2024), "market_ids"
        ]
    )

    DATASET = pyblp.MicroDataset(
        name="US_2015_2024_pooled",
        observations=N_observed,
        compute_weights=lambda t, p, a: (
            np.ones((a.size, 1 + p.size))
            if t in valid_markets
            else np.zeros((a.size, 1 + p.size))
        ),
    )

    # --- pooled raw incomes across valid markets ---
    all_incomes = []
    for t in sorted(valid_markets):
        a_t = agent_data[agent_data["market_ids"] == t]
        all_incomes.append(a_t["income_raw_10k"].values)
    all_incomes = np.concatenate(all_incomes) if all_incomes else np.array([])

    if all_incomes.size == 0:
        raise ValueError("No income observations found after aligning markets.")

    q20_raw, q40_raw, q60_raw, q80_raw = np.quantile(
        all_incomes, [0.2, 0.4, 0.6, 0.8]
    )

    quintile_cutoffs_raw = (q20_raw, q40_raw, q60_raw, q80_raw)

    def inc_in_quintile(a, k):
        """Return 1{income in quintile k} using log income demographics."""
        log_inc = a.demographics[:, INCOME_COL]
        inc_raw_10k = np.exp(log_inc)

        if k == 1:
            mask = inc_raw_10k <= q20_raw
        elif k == 2:
            mask = (inc_raw_10k > q20_raw) & (inc_raw_10k <= q40_raw)
        elif k == 3:
            mask = (inc_raw_10k > q40_raw) & (inc_raw_10k <= q60_raw)
        elif k == 4:
            mask = (inc_raw_10k > q60_raw) & (inc_raw_10k <= q80_raw)
        elif k == 5:
            mask = inc_raw_10k > q80_raw
        else:
            raise ValueError("k must be 1..5")
        return mask.astype(float)

    def product_vec_inside(p):
        """[0, 1,1,...,1] – sums over *inside* options only (j>0)."""
        return np.r_[0.0, np.ones(p.size, dtype=float)]

    def product_vec_all(p):
        """[1, 1,1,...,1] – sums over outside+inside; equals agent-only expectation."""
        return np.r_[1.0, np.ones(p.size, dtype=float)]

    # --- MicroParts for purchase prob by quintile ---

    def mk_num_Qk(k):
        return pyblp.MicroPart(
            name=f"E[1{{Q{k}}} * 1{{j>0}}]",
            dataset=DATASET,
            compute_values=lambda t, p, a, k=k: np.outer(
                inc_in_quintile(a, k), product_vec_inside(p)
            ),
        )

    def mk_den_Qk(k):
        return pyblp.MicroPart(
            name=f"E[1{{Q{k}}}]",
            dataset=DATASET,
            compute_values=lambda t, p, a, k=k: np.outer(
                inc_in_quintile(a, k), product_vec_all(p)
            ),
        )

    num_Q1, den_Q1 = mk_num_Qk(1), mk_den_Qk(1)
    num_Q2, den_Q2 = mk_num_Qk(2), mk_den_Qk(2)
    num_Q3, den_Q3 = mk_num_Qk(3), mk_den_Qk(3)
    num_Q4, den_Q4 = mk_num_Qk(4), mk_den_Qk(4)
    num_Q5, den_Q5 = mk_num_Qk(5), mk_den_Qk(5)

    # mappings
    def two_ratio(v):
        num_k, den_k, num_1, den_1 = v
        pk = num_k / den_k
        p1 = num_1 / den_1
        return pk / p1

    def two_ratio_grad(v):
        num_k, den_k, num_1, den_1 = v
        pk = num_k / den_k
        p1 = num_1 / den_1

        dpk_dnumk = 1.0 / den_k
        dpk_ddenk = -num_k / (den_k**2)

        dp1_dnum1 = 1.0 / den_1
        dp1_dden1 = -num_1 / (den_1**2)

        dr_dnumk = (1.0 / p1) * dpk_dnumk
        dr_ddenk = (1.0 / p1) * dpk_ddenk
        dr_dnum1 = -(pk / (p1**2)) * dp1_dnum1
        dr_dden1 = -(pk / (p1**2)) * dp1_dden1
        return [dr_dnumk, dr_ddenk, dr_dnum1, dr_dden1]

    Q2Q1_target = 2.0222
    Q3Q1_target = 2.8043
    Q4Q1_target = 3.4695
    Q5Q1_target = 5.8406

    micro_moments_income_quintiles = [
        pyblp.MicroMoment(
            name="P(purchase|Q2)/P(purchase|Q1), pooled 2015–2024",
            value=Q2Q1_target,
            parts=[num_Q2, den_Q2, num_Q1, den_Q1],
            compute_value=two_ratio,
            compute_gradient=two_ratio_grad,
        ),
        pyblp.MicroMoment(
            name="P(purchase|Q3)/P(purchase|Q1), pooled 2015–2024",
            value=Q3Q1_target,
            parts=[num_Q3, den_Q3, num_Q1, den_Q1],
            compute_value=two_ratio,
            compute_gradient=two_ratio_grad,
        ),
        pyblp.MicroMoment(
            name="P(purchase|Q4)/P(purchase|Q1), pooled 2015–2024",
            value=Q4Q1_target,
            parts=[num_Q4, den_Q4, num_Q1, den_Q1],
            compute_value=two_ratio,
            compute_gradient=two_ratio_grad,
        ),
        pyblp.MicroMoment(
            name="P(purchase|Q5)/P(purchase|Q1), pooled 2015–2024",
            value=Q5Q1_target,
            parts=[num_Q5, den_Q5, num_Q1, den_Q1],
            compute_value=two_ratio,
            compute_gradient=two_ratio_grad,
        ),
    ]

    ##############################
    # CEX Micro moments - PRICES
    ##############################

    pre_diff_targets = {
        2: 1700.803676 / 1e5,
        3: 1597.430281 / 1e5,
        4: 562.274069 / 1e5,
        5: 3219.740804 / 1e5,
    }

    cov_diff_targets = {
        2: 537.881715 / 1e5,
        3: 54.898609 / 1e5,
        4: 3292.777063 / 1e5,
        5: 12913.054072 / 1e5,
    }

    pool_diff_targets = {
        2: 1101.814016 / 1e5,
        3: 1662.925492 / 1e5,
        4: 1145.179107 / 1e5,
        5: 6256.424324 / 1e5,
    }

    pre_years = {2015, 2016, 2017, 2018, 2019}
    cov_years = {2020, 2021, 2022}
    pooled_years = set(range(2015, 2024))

    prod_years = set(np.unique(product_data["market_ids"]))
    agent_years = set(np.unique(agent_data["market_ids"]))

    def intersect_years(candidate_years):
        years = {y for y in candidate_years if (y in prod_years and y in agent_years)}
        if not years:
            raise ValueError(f"No overlapping markets in candidate years {candidate_years}.")
        return years

    pre_years = intersect_years(pre_years)
    cov_years = intersect_years(cov_years)
    pooled_years = intersect_years(pooled_years)

    Pre1519 = pyblp.MicroDataset(
        name="US_2015_2019_price_diff",
        observations=N_pre,
        market_ids=sorted(pre_years),
        compute_weights=lambda t, p, a: np.ones((a.size, 1 + p.size)),
    )

    Cov2022 = pyblp.MicroDataset(
        name="US_2020_2022_price_diff",
        observations=N_cov,
        market_ids=sorted(cov_years),
        compute_weights=lambda t, p, a: np.ones((a.size, 1 + p.size)),
    )

    Pool1523 = pyblp.MicroDataset(
        name="US_2015_2023_price_diff",
        observations=N_pool,
        market_ids=sorted(pooled_years),
        compute_weights=lambda t, p, a: np.ones((a.size, 1 + p.size)),
    )

    def price_vec_inside(p):
        prices = np.asarray(p.prices).reshape(-1)
        return np.r_[0.0, prices]

    def mk_price_diff_parts(dataset):
        def num_Qk(k):
            return pyblp.MicroPart(
                name=f"E[1{{Q{k}}}*1{{j>0}}*price]({dataset.name})",
                dataset=dataset,
                compute_values=lambda t, p, a, k=k: np.outer(
                    inc_in_quintile(a, k), price_vec_inside(p)
                ),
            )

        def den_Qk(k):
            return pyblp.MicroPart(
                name=f"E[1{{Q{k}}}*1{{j>0}}]({dataset.name})",
                dataset=dataset,
                compute_values=lambda t, p, a, k=k: np.outer(
                    inc_in_quintile(a, k), product_vec_inside(p)
                ),
            )

        return {k: (num_Qk(k), den_Qk(k)) for k in [1, 2, 3, 4, 5]}

    parts_pre = mk_price_diff_parts(Pre1519)
    parts_cov = mk_price_diff_parts(Cov2022)
    parts_pool = mk_price_diff_parts(Pool1523)

    def diff_two_cond_means(v):
        num_k, den_k, num_1, den_1 = v
        return (num_k / den_k) - (num_1 / den_1)

    def diff_two_cond_means_grad(v):
        num_k, den_k, num_1, den_1 = v
        dmk = [1.0 / den_k, -num_k / (den_k**2)]
        dm1 = [1.0 / den_1, -num_1 / (den_1**2)]
        return np.array([dmk[0], dmk[1], -dm1[0], -dm1[1]], dtype=float)

    micro_price_diff_pre = [
        pyblp.MicroMoment(
            name=f"Mean price(Q{k}) - Mean price(Q1) | 2015–2019",
            value=pre_diff_targets[k],
            parts=[
                parts_pre[k][0],
                parts_pre[k][1],
                parts_pre[1][0],
                parts_pre[1][1],
            ],
            compute_value=diff_two_cond_means,
            compute_gradient=diff_two_cond_means_grad,
        )
        for k in [2, 3, 4, 5]
    ]

    micro_price_diff_cov = [
        pyblp.MicroMoment(
            name=f"Mean price(Q{k}) - Mean price(Q1) | 2020–2022",
            value=cov_diff_targets[k],
            parts=[
                parts_cov[k][0],
                parts_cov[k][1],
                parts_cov[1][0],
                parts_cov[1][1],
            ],
            compute_value=diff_two_cond_means,
            compute_gradient=diff_two_cond_means_grad,
        )
        for k in [2, 3, 4, 5]
    ]

    micro_price_diff_pool = [
        pyblp.MicroMoment(
            name=f"Mean price(Q{k}) - Mean price(Q1) | 2015–2023",
            value=pool_diff_targets[k],
            parts=[
                parts_pool[k][0],
                parts_pool[k][1],
                parts_pool[1][0],
                parts_pool[1][1],
            ],
            compute_value=diff_two_cond_means,
            compute_gradient=diff_two_cond_means_grad,
        )
        for k in [2, 3, 4, 5]
    ]

    ##############################
    # CEX Micro moments - AGE × PRICE (2015)
    ##############################

    age_price_diff_targets_2015 = {
        ">60": 0.0257,
        "50_60": 0.0239,
        "40_50": 0.0265,
        "30_40": 0.0265,
    }

    AGE_COL = age_col

    N_age_2015 = 1000
    markets_2015 = set(
        product_data.loc[product_data["market_ids"] == 2015, "market_ids"]
    )

    AgePrice2015 = pyblp.MicroDataset(
        name="US_2015_age_price_diff",
        observations=N_age_2015,
        market_ids=2015,
        compute_weights=lambda t, p, a: (
            np.ones((a.size, 1 + p.size), dtype=float)
            if t in markets_2015
            else np.zeros((a.size, 1 + p.size), dtype=float)
        ),
    )

    def _age_from_demog(a):
        return a.demographics[:, AGE_COL]

    def age_lt_30(a):
        return (_age_from_demog(a) < 30).astype(float)

    def age_30_40(a):
        age = _age_from_demog(a)
        return ((age >= 30) & (age < 40)).astype(float)

    def age_40_50(a):
        age = _age_from_demog(a)
        return ((age >= 40) & (age < 50)).astype(float)

    def age_50_60(a):
        age = _age_from_demog(a)
        return ((age >= 50) & (age < 60)).astype(float)

    def age_gt_60(a):
        return (_age_from_demog(a) > 60).astype(float)

    def mk_age_price_parts(dataset, age_indicator_func, label):
        num = pyblp.MicroPart(
            name=f"E[1{{{label}}} * 1{{j>0}} * price]({dataset.name})",
            dataset=dataset,
            compute_values=lambda t, p, a: np.outer(
                age_indicator_func(a), price_vec_inside(p)
            ),
        )
        den = pyblp.MicroPart(
            name=f"E[1{{{label}}} * 1{{j>0}}]({dataset.name})",
            dataset=dataset,
            compute_values=lambda t, p, a: np.outer(
                age_indicator_func(a), product_vec_inside(p)
            ),
        )
        return num, den

    num_age_lt30_2015, den_age_lt30_2015 = mk_age_price_parts(
        AgePrice2015, age_lt_30, "Age<30"
    )
    num_age_30_40_2015, den_age_30_40_2015 = mk_age_price_parts(
        AgePrice2015, age_30_40, "Age30-40"
    )
    num_age_40_50_2015, den_age_40_50_2015 = mk_age_price_parts(
        AgePrice2015, age_40_50, "Age40-50"
    )
    num_age_50_60_2015, den_age_50_60_2015 = mk_age_price_parts(
        AgePrice2015, age_50_60, "Age50-60"
    )
    num_age_gt60_2015, den_age_gt60_2015 = mk_age_price_parts(
        AgePrice2015, age_gt_60, "Age>60"
    )

    micro_age_price_diff_2015 = [
        pyblp.MicroMoment(
            name="E[price | Age40-50] - E[price | Age<30] | 2015",
            value=age_price_diff_targets_2015["40_50"],
            parts=[
                num_age_40_50_2015,
                den_age_40_50_2015,
                num_age_lt30_2015,
                den_age_lt30_2015,
            ],
            compute_value=diff_two_cond_means,
            compute_gradient=diff_two_cond_means_grad,
        ),
        pyblp.MicroMoment(
            name="E[price | Age30-40] - E[price | Age<30] | 2015",
            value=age_price_diff_targets_2015["30_40"],
            parts=[
                num_age_30_40_2015,
                den_age_30_40_2015,
                num_age_lt30_2015,
                den_age_lt30_2015,
            ],
            compute_value=diff_two_cond_means,
            compute_gradient=diff_two_cond_means_grad,
        ),
    ]

    return (
        micro_moments_income_quintiles,
        micro_price_diff_pre,
        micro_price_diff_cov,
        micro_price_diff_pool,
        micro_age_price_diff_2015,
        quintile_cutoffs_raw,
    )
