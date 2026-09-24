"""
src/features/rfm.py
--------------------
Customer-level feature engineering and descriptive RFM scoring.

All calculations use ONLY information on or before SNAPSHOT_DATE.
"""

import numpy as np
import pandas as pd

from src.config import SNAPSHOT_DATE


# ---------------------------------------------------------------------------
# Customer-level feature table
# ---------------------------------------------------------------------------


def compute_customer_features(
    df_obs: pd.DataFrame, snapshot: pd.Timestamp = SNAPSHOT_DATE
) -> pd.DataFrame:
    """Build the customer-level analytical feature table.

    Parameters
    ----------
    df_obs : pd.DataFrame
        Qualifying-purchase rows limited to the observation period
        (InvoiceDate <= snapshot).  Must contain Revenue column.
    snapshot : pd.Timestamp
        Snapshot date.

    Returns
    -------
    pd.DataFrame indexed by CustomerID with columns:
        Recency, Frequency, Monetary, AvgOrderValue,
        UniqueProducts, TotalItems, ActiveMonths,
        TenureDays, PurchaseSpanDays, Country
    """
    # Use date-level comparison so all intraday transactions on the snapshot
    # date are included (snapshot = 2011-08-31 means all of that calendar day).
    snap_date = snapshot.date()
    df = df_obs[df_obs["InvoiceDate"].dt.date <= snap_date].copy()

    agg = df.groupby("CustomerID").agg(
        last_purchase=("InvoiceDate", "max"),
        first_purchase=("InvoiceDate", "min"),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("Revenue", "sum"),
        UniqueProducts=("StockCode", "nunique"),
        TotalItems=("Quantity", "sum"),
    )

    # Active months: distinct year-month combinations
    monthly = (
        df.assign(YearMonth=df["InvoiceDate"].dt.to_period("M"))
        .groupby("CustomerID")["YearMonth"]
        .nunique()
        .rename("ActiveMonths")
    )

    agg = agg.join(monthly)

    # Normalize datetimes to midnight before computing day differences.
    # This prevents intraday timestamps on the snapshot date from producing
    # negative recency or off-by-one tenure values.
    snap_ts = pd.Timestamp(snap_date)  # midnight of snapshot date
    last_purchase_date = agg["last_purchase"].dt.normalize()
    first_purchase_date = agg["first_purchase"].dt.normalize()

    agg["Recency"] = (snap_ts - last_purchase_date).dt.days
    agg["TenureDays"] = (snap_ts - first_purchase_date).dt.days
    agg["PurchaseSpanDays"] = (last_purchase_date - first_purchase_date).dt.days
    agg["AvgOrderValue"] = agg["Monetary"] / agg["Frequency"]

    # Country: most frequent; tie-break by most recent purchase
    country = _resolve_country(df)
    agg = agg.join(country)

    feature_cols = [
        "Recency",
        "Frequency",
        "Monetary",
        "AvgOrderValue",
        "UniqueProducts",
        "TotalItems",
        "ActiveMonths",
        "TenureDays",
        "PurchaseSpanDays",
        "Country",
    ]
    return agg[feature_cols].copy()


def _resolve_country(df: pd.DataFrame) -> pd.Series:
    """Return most frequent Country per customer; tie-break by most-recent purchase."""
    # Most frequent country
    freq = (
        df.groupby(["CustomerID", "Country"])["InvoiceNo"]
        .count()
        .rename("cnt")
        .reset_index()
    )
    # Most recent transaction date per customer-country
    recent = (
        df.groupby(["CustomerID", "Country"])["InvoiceDate"]
        .max()
        .rename("last_date")
        .reset_index()
    )
    merged = freq.merge(recent, on=["CustomerID", "Country"])
    # Sort: descending cnt, descending last_date → pick first per customer
    merged = merged.sort_values(
        ["CustomerID", "cnt", "last_date"], ascending=[True, False, False]
    )
    result = (
        merged.groupby("CustomerID")["Country"].first().rename("Country")
    )
    return result


# ---------------------------------------------------------------------------
# Descriptive RFM scoring
# ---------------------------------------------------------------------------


def _quintile_score(series: pd.Series, ascending: bool = True) -> pd.Series:
    """Assign quintile scores 1–5 to a Series with a deterministic fallback.

    Parameters
    ----------
    series : pd.Series
        Numeric values to score.
    ascending : bool
        If True, higher values → higher scores.
        If False (Recency), lower values → higher scores.

    Uses pd.qcut where possible; falls back to rank-based quantile
    assignment when qcut raises a duplicate-bin ValueError (ties).
    """
    labels = [1, 2, 3, 4, 5] if ascending else [5, 4, 3, 2, 1]

    try:
        scores = pd.qcut(series, q=5, labels=labels, duplicates="drop")
        # If duplicates="drop" produces fewer than 5 bins, scores may be NaN
        # for edge cases; fall through to rank-based approach if so.
        if scores.isna().any():
            raise ValueError("qcut produced NaN scores — using rank fallback")
        return scores.astype(int)
    except (ValueError, IndexError):
        pass

    # Rank-based fallback: deterministic, handles ties
    ranked = series.rank(method="first", ascending=ascending)
    n = len(series)
    # Map rank to quintile 1–5
    quintile = pd.cut(
        ranked,
        bins=np.linspace(0, n + 1, 6),
        labels=[1, 2, 3, 4, 5],
        include_lowest=True,
        right=True,
    )
    return quintile.astype(int)


def add_rfm_scores(features: pd.DataFrame) -> pd.DataFrame:
    """Add quintile-based R, F, M scores and combined RFM_Score to feature table.

    Scoring rules:
      R: lower recency → better → best quintile = 5  (R=5 is most recent)
      F: higher frequency → better → highest quintile = 5
      M: higher monetary → better → highest quintile = 5
      RFM_Score = R + F + M  (range 3–15)

    Segment labels:
      13–15 → Very High Value
      10–12 → High Value
       7–9  → Medium Value
       3–6  → Lower Value

    Note: this is descriptive segmentation only.
    Scored from the full eligible population — NOT fitted on a training split.
    Column names R, F, M (not R_Score, F_Score, M_Score) match the
    customer_features.csv schema.
    """
    df = features.copy()

    # R: lower recency is better → invert labels so lowest recency → score 5
    df["R"] = _quintile_score(df["Recency"], ascending=False)
    df["F"] = _quintile_score(df["Frequency"], ascending=True)
    df["M"] = _quintile_score(df["Monetary"], ascending=True)

    df["RFM_Score"] = df["R"] + df["F"] + df["M"]
    df["RFM_Segment"] = df["RFM_Score"].map(_rfm_label)

    return df


def _rfm_label(score: int) -> str:
    if score >= 13:
        return "Very High Value"
    elif score >= 10:
        return "High Value"
    elif score >= 7:
        return "Medium Value"
    else:
        return "Lower Value"
