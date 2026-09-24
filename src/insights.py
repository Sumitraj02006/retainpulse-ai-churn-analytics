"""
src/insights.py
----------------
Reusable, evidence-based business insight calculations.

Functions here derive operational outputs from the customer feature table
and model predictions.  They never invent values — all outputs are computed
from actual data passed in.
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Risk tier assignment
# ---------------------------------------------------------------------------

RISK_TIER_THRESHOLDS = {
    "high_percentile": 80,    # top 20% → High Risk
    "medium_percentile": 40,  # 40th–80th → Medium Risk; below 40th → Low Risk
}


def assign_risk_tiers(churn_probabilities: pd.Series) -> pd.Series:
    """Assign operational risk tiers based on predicted churn probability.

    Tiers (based on percentile of predicted probability distribution):
      Top 20%  → High Risk
      Middle 40% (40th–80th pct) → Medium Risk
      Bottom 40% → Low Risk

    Parameters
    ----------
    churn_probabilities : pd.Series
        Per-customer churn probability (0–1), indexed by CustomerID.

    Returns
    -------
    pd.Series
        Risk tier string per customer, same index.
    """
    high_thresh = np.percentile(churn_probabilities, RISK_TIER_THRESHOLDS["high_percentile"])
    med_thresh = np.percentile(churn_probabilities, RISK_TIER_THRESHOLDS["medium_percentile"])

    def _tier(p):
        if p >= high_thresh:
            return "High Risk"
        elif p >= med_thresh:
            return "Medium Risk"
        else:
            return "Low Risk"

    return churn_probabilities.map(_tier)


# ---------------------------------------------------------------------------
# Retention priority matrix
# ---------------------------------------------------------------------------

_PRIORITY_MAP = {
    ("High Risk", "Very High Value"): "Priority retention attention",
    ("High Risk", "High Value"): "Priority retention attention",
    ("High Risk", "Medium Value"): "Targeted attention",
    ("High Risk", "Lower Value"): "Lower-cost intervention",
    ("Medium Risk", "Very High Value"): "Monitor / targeted engagement",
    ("Medium Risk", "High Value"): "Monitor / targeted engagement",
    ("Medium Risk", "Medium Value"): "Standard engagement",
    ("Medium Risk", "Lower Value"): "Standard engagement",
    ("Low Risk", "Very High Value"): "Relationship / loyalty attention",
    ("Low Risk", "High Value"): "Relationship / loyalty attention",
    ("Low Risk", "Medium Value"): "Standard engagement",
    ("Low Risk", "Lower Value"): "Standard engagement",
}


def assign_retention_priority(risk_tier: str, rfm_segment: str) -> str:
    """Return evidence-based retention priority for a customer.

    Parameters
    ----------
    risk_tier : str
        One of 'High Risk', 'Medium Risk', 'Low Risk'.
    rfm_segment : str
        One of 'Very High Value', 'High Value', 'Medium Value', 'Lower Value'.

    Returns
    -------
    str
        Retention priority label.
    """
    return _PRIORITY_MAP.get(
        (risk_tier, rfm_segment),
        "Standard engagement",
    )


def build_customer_risk_table(
    features: pd.DataFrame,
    churn_probabilities: pd.Series,
) -> pd.DataFrame:
    """Combine features, probabilities, tiers, and retention priorities.

    Parameters
    ----------
    features : pd.DataFrame
        Customer feature table with RFM scores/segments (indexed by CustomerID).
    churn_probabilities : pd.Series
        Model-output probabilities (indexed by CustomerID).

    Returns
    -------
    pd.DataFrame
        One row per customer with all analytical columns plus:
        Churn_Probability (%), Risk_Tier, Retention_Priority.
    """
    df = features.copy()
    df = df.join(churn_probabilities.rename("Churn_Probability_Raw"), how="left")
    df["Churn_Probability"] = (df["Churn_Probability_Raw"] * 100).round(2)
    df["Risk_Tier"] = assign_risk_tiers(df["Churn_Probability_Raw"])

    if "RFM_Segment" in df.columns:
        df["Retention_Priority"] = df.apply(
            lambda row: assign_retention_priority(row["Risk_Tier"], row["RFM_Segment"]),
            axis=1,
        )
    return df.drop(columns=["Churn_Probability_Raw"])


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------


def churn_rate_by_group(df: pd.DataFrame, group_col: str, churn_col: str = "Churn_90D") -> pd.DataFrame:
    """Return inactivity rate by a categorical grouping column."""
    return (
        df.groupby(group_col)[churn_col]
        .agg(n="count", churned="sum")
        .assign(churn_rate=lambda x: x["churned"] / x["n"])
        .reset_index()
    )
