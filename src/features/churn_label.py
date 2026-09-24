"""
src/features/churn_label.py
-----------------------------
Generate the project-defined Churn_90D target.

IMPORTANT: The future window (FUTURE_START – FUTURE_END) is used ONLY to
assign Churn_90D.  It must never feed into predictor features.
"""

import pandas as pd

from src.config import FUTURE_END, FUTURE_START, SNAPSHOT_DATE


def generate_churn_label(
    df_all_qualified: pd.DataFrame,
    eligible_customer_ids,
    future_start: pd.Timestamp = FUTURE_START,
    future_end: pd.Timestamp = FUTURE_END,
) -> pd.DataFrame:
    """Create Churn_90D for each eligible customer.

    Churn_90D = 1 : no qualifying purchase in [future_start, future_end]
    Churn_90D = 0 : at least one qualifying purchase in [future_start, future_end]

    Parameters
    ----------
    df_all_qualified : pd.DataFrame
        ALL qualifying-purchase rows (observation + future window combined).
    eligible_customer_ids : array-like
        CustomerIDs that passed the eligibility filter.
    future_start, future_end : pd.Timestamp
        Boundaries of the future labelling window (inclusive).

    Returns
    -------
    pd.DataFrame
        One row per eligible customer with columns [CustomerID, Churn_90D].
    """
    # Future-window purchases for eligible customers only.
    # Use date-level comparison so all intraday transactions on the boundary
    # dates are correctly included/excluded.
    _dates = df_all_qualified["InvoiceDate"].dt.date
    future_mask = (
        (_dates >= future_start.date())
        & (_dates <= future_end.date())
        & df_all_qualified["CustomerID"].isin(eligible_customer_ids)
    )
    future_buyers = set(
        df_all_qualified.loc[future_mask, "CustomerID"].unique()
    )

    labels = pd.DataFrame({"CustomerID": list(eligible_customer_ids)})
    labels["Churn_90D"] = labels["CustomerID"].apply(
        lambda cid: 0 if cid in future_buyers else 1
    )
    return labels.set_index("CustomerID")
