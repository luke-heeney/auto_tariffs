import numpy as np
import pandas as pd
import pyblp


def build_income_price_moments(
    product_data: pd.DataFrame,
    income_col: int = 0,
    N_mri: int = 1000,
):
    """
    Build MRI 2019 income × price micro-moments.

    Parameters
    ----------
    product_data : DataFrame
        Product-level BLP data with a 'market_ids' column.
    income_col : int, default 0
        Column index of log income in pyblp AgentData demographics.
    N_mri : int, default 1000
        Effective micro sample size.

    Returns
    -------
    list[pyblp.MicroMoment]
        The list `micro_moments_income_price`.
    """

    # ---- dataset ----
    markets_2019_us = set(
        product_data.loc[product_data["market_ids"] == 2019, "market_ids"]
    )

    MRI_US_2019 = pyblp.MicroDataset(
        name="MRI_Simmons_2019_US",
        observations=N_mri,
        market_ids=2019,
        compute_weights=lambda t, p, a: (
            np.ones((a.size, 1 + p.size))
            if t in markets_2019_us
            else np.zeros((a.size, 1 + p.size))
        ),
    )

    INCOME_COL = income_col

    # ---- helpers ----
    def _income_raw_from_demog(a):
        """Convert log income demographics back to raw income in $10k units."""
        log_inc = a.demographics[:, INCOME_COL]
        return np.exp(log_inc)

    def price_gt_threshold_vector(p, threshold=0.5, include_outside=True):
        prices_1d = np.asarray(p.prices).reshape(-1)
        flag_J = (prices_1d > threshold).astype(float)
        return np.concatenate(([0.0], flag_J)) if include_outside else flag_J

    def income_indicator(a, *, gt=None, lo=None, hi=None):
        """
        Build agent-side indicator using raw income (in $10k units).
        Thresholds should be specified in the same units.
        """
        inc_raw = _income_raw_from_demog(a)

        if gt is not None:
            return (inc_raw > gt).astype(float)
        if lo is not None and hi is not None:
            return ((inc_raw >= lo) & (inc_raw <= hi)).astype(float)
        raise ValueError("Set gt=... or lo=... and hi=...")

    # ---- MicroParts ----

    num_inc_gt_100k = pyblp.MicroPart(
        name="E[1{inc>100k} * 1{j>0} * 1{price>50k}]",
        dataset=MRI_US_2019,
        compute_values=lambda t, p, a: np.outer(
            income_indicator(a, gt=9.271),  # income > ~$100k (raw units in 10k)
            price_gt_threshold_vector(p, 0.4635),
        ),
    )

    num_inc_60_100k = pyblp.MicroPart(
        name="E[1{60k<=inc<=100k} * 1{j>0} * 1{price>50k}]",
        dataset=MRI_US_2019,
        compute_values=lambda t, p, a: np.outer(
            income_indicator(a, lo=5.563, hi=9.271),
            price_gt_threshold_vector(p, 0.4635),
        ),
    )

    den_price_gt_50k = pyblp.MicroPart(
        name="E[1{j>0} * 1{price>50k}]",
        dataset=MRI_US_2019,
        compute_values=lambda t, p, a: np.outer(
            np.ones(a.size),
            price_gt_threshold_vector(p, 0.4635),
        ),
    )

    # ---- MicroMoments ----

    ratio = lambda v: v[0] / v[1]
    ratio_grad = lambda v: [1.0 / v[1], -v[0] / (v[1] ** 2)]

    micro_moments_income_price = [
        pyblp.MicroMoment(
            name="P(inc>100k | price>50k, purchase)",
            value=0.631,
            parts=[num_inc_gt_100k, den_price_gt_50k],
            compute_value=ratio,
            compute_gradient=ratio_grad,
        ),
        pyblp.MicroMoment(
            name="P(60k<=inc<=100k | price>50k, purchase)",
            value=0.212,
            parts=[num_inc_60_100k, den_price_gt_50k],
            compute_value=ratio,
            compute_gradient=ratio_grad,
        ),
    ]

    return micro_moments_income_price
