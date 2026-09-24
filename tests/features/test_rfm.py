"""
tests/features/test_rfm.py
---------------------------
Tests for src/features/rfm.py — Recency, Frequency, Monetary calculations
and RFM scoring.
"""

import pandas as pd
import pytest

from src.features.rfm import add_rfm_scores, compute_customer_features
from src.config import SNAPSHOT_DATE


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


def _make_obs():
    """Minimal qualifying-purchase observation DataFrame for two customers."""
    rows = [
        # Customer 100: 3 invoices over ~6 months, last purchase 2011-07-01
        {"CustomerID": 100.0, "InvoiceNo": "A1", "StockCode": "X1",
         "Quantity": 2, "UnitPrice": 10.0, "Revenue": 20.0,
         "InvoiceDate": pd.Timestamp("2011-01-15"), "Country": "United Kingdom"},
        {"CustomerID": 100.0, "InvoiceNo": "A2", "StockCode": "X2",
         "Quantity": 1, "UnitPrice": 5.0, "Revenue": 5.0,
         "InvoiceDate": pd.Timestamp("2011-04-10"), "Country": "United Kingdom"},
        {"CustomerID": 100.0, "InvoiceNo": "A3", "StockCode": "X1",
         "Quantity": 3, "UnitPrice": 10.0, "Revenue": 30.0,
         "InvoiceDate": pd.Timestamp("2011-07-01"), "Country": "United Kingdom"},
        # Customer 200: 2 invoices, last purchase 2011-06-01
        {"CustomerID": 200.0, "InvoiceNo": "B1", "StockCode": "Y1",
         "Quantity": 5, "UnitPrice": 4.0, "Revenue": 20.0,
         "InvoiceDate": pd.Timestamp("2011-02-01"), "Country": "Germany"},
        {"CustomerID": 200.0, "InvoiceNo": "B2", "StockCode": "Y2",
         "Quantity": 2, "UnitPrice": 8.0, "Revenue": 16.0,
         "InvoiceDate": pd.Timestamp("2011-06-01"), "Country": "Germany"},
    ]
    return pd.DataFrame(rows)


def _make_obs_large():
    """Ten-customer dataset — large enough for robust quintile scoring without fallback."""
    import random
    random.seed(42)
    rows = []
    base = pd.Timestamp("2011-01-01")
    for cid in range(10):
        n_inv = cid + 2
        for i in range(n_inv):
            rows.append({
                "CustomerID": float(cid + 1),
                "InvoiceNo": f"INV_{cid}_{i}",
                "StockCode": f"SC{i}",
                "Quantity": i + 1,
                "UnitPrice": float((i + 1) * 2),
                "Revenue": float((i + 1) ** 2 * 2),
                "InvoiceDate": base + pd.Timedelta(days=i * 30 + cid * 5),
                "Country": "United Kingdom",
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Observation-period filtering
# ---------------------------------------------------------------------------


def test_observation_filter_excludes_post_snapshot():
    """Rows with InvoiceDate > SNAPSHOT_DATE must not affect features."""
    rows = [
        # Customer 100 — two obs-period invoices
        {"CustomerID": 100.0, "InvoiceNo": "A1", "StockCode": "X1",
         "Quantity": 1, "UnitPrice": 10.0, "Revenue": 10.0,
         "InvoiceDate": pd.Timestamp("2011-01-01"), "Country": "UK"},
        {"CustomerID": 100.0, "InvoiceNo": "A2", "StockCode": "X2",
         "Quantity": 1, "UnitPrice": 10.0, "Revenue": 10.0,
         "InvoiceDate": pd.Timestamp("2011-06-01"), "Country": "UK"},
        # Post-snapshot row — must be excluded from features
        {"CustomerID": 100.0, "InvoiceNo": "A3", "StockCode": "X3",
         "Quantity": 100, "UnitPrice": 999.0, "Revenue": 99900.0,
         "InvoiceDate": pd.Timestamp("2011-09-15"), "Country": "UK"},
    ]
    df = pd.DataFrame(rows)
    features = compute_customer_features(df, snapshot=SNAPSHOT_DATE)
    # Monetary must NOT include the post-snapshot row
    assert pytest.approx(features.loc[100.0, "Monetary"]) == 20.0
    # Frequency must NOT include the post-snapshot invoice
    assert features.loc[100.0, "Frequency"] == 2
    # Recency must be based on last obs-period purchase (2011-06-01), not 2011-09-15
    expected_recency = (SNAPSHOT_DATE - pd.Timestamp("2011-06-01")).days
    assert features.loc[100.0, "Recency"] == expected_recency


# ---------------------------------------------------------------------------
# RFM feature calculations
# ---------------------------------------------------------------------------


def test_recency_calculation():
    df = _make_obs()
    features = compute_customer_features(df, snapshot=SNAPSHOT_DATE)
    # Customer 100 last purchase 2011-07-01 → recency = (2011-08-31 - 2011-07-01).days = 61
    assert features.loc[100.0, "Recency"] == 61


def test_frequency_calculation():
    df = _make_obs()
    features = compute_customer_features(df, snapshot=SNAPSHOT_DATE)
    assert features.loc[100.0, "Frequency"] == 3
    assert features.loc[200.0, "Frequency"] == 2


def test_monetary_calculation():
    df = _make_obs()
    features = compute_customer_features(df, snapshot=SNAPSHOT_DATE)
    assert pytest.approx(features.loc[100.0, "Monetary"]) == 55.0  # 20+5+30
    assert pytest.approx(features.loc[200.0, "Monetary"]) == 36.0  # 20+16


def test_avg_order_value():
    df = _make_obs()
    features = compute_customer_features(df, snapshot=SNAPSHOT_DATE)
    assert pytest.approx(features.loc[100.0, "AvgOrderValue"]) == 55.0 / 3


def test_unique_products():
    df = _make_obs()
    features = compute_customer_features(df, snapshot=SNAPSHOT_DATE)
    # Customer 100 has StockCodes X1, X2 → 2 unique
    assert features.loc[100.0, "UniqueProducts"] == 2


def test_total_items():
    df = _make_obs()
    features = compute_customer_features(df, snapshot=SNAPSHOT_DATE)
    # Customer 100: quantities 2+1+3 = 6
    assert features.loc[100.0, "TotalItems"] == 6


def test_active_months():
    """ActiveMonths counts distinct calendar months with purchases."""
    rows = [
        # Jan, Mar, Jul — 3 distinct months
        {"CustomerID": 100.0, "InvoiceNo": "A1", "StockCode": "X1",
         "Quantity": 1, "UnitPrice": 1.0, "Revenue": 1.0,
         "InvoiceDate": pd.Timestamp("2011-01-10"), "Country": "UK"},
        {"CustomerID": 100.0, "InvoiceNo": "A2", "StockCode": "X1",
         "Quantity": 1, "UnitPrice": 1.0, "Revenue": 1.0,
         "InvoiceDate": pd.Timestamp("2011-01-25"), "Country": "UK"},  # same month
        {"CustomerID": 100.0, "InvoiceNo": "A3", "StockCode": "X1",
         "Quantity": 1, "UnitPrice": 1.0, "Revenue": 1.0,
         "InvoiceDate": pd.Timestamp("2011-03-05"), "Country": "UK"},
        {"CustomerID": 100.0, "InvoiceNo": "A4", "StockCode": "X1",
         "Quantity": 1, "UnitPrice": 1.0, "Revenue": 1.0,
         "InvoiceDate": pd.Timestamp("2011-07-01"), "Country": "UK"},
    ]
    df = pd.DataFrame(rows)
    features = compute_customer_features(df, snapshot=SNAPSHOT_DATE)
    assert features.loc[100.0, "ActiveMonths"] == 3


def test_tenure_days():
    df = _make_obs()
    features = compute_customer_features(df, snapshot=SNAPSHOT_DATE)
    # Customer 100 first purchase 2011-01-15 → tenure = (2011-08-31 - 2011-01-15).days = 228
    expected = (SNAPSHOT_DATE - pd.Timestamp("2011-01-15")).days
    assert features.loc[100.0, "TenureDays"] == expected


def test_purchase_span_days():
    df = _make_obs()
    features = compute_customer_features(df, snapshot=SNAPSHOT_DATE)
    # Customer 100: last 2011-07-01, first 2011-01-15 → 167 days
    expected = (pd.Timestamp("2011-07-01") - pd.Timestamp("2011-01-15")).days
    assert features.loc[100.0, "PurchaseSpanDays"] == expected


def test_country_most_frequent():
    """Country resolves to the most frequently occurring value."""
    rows = [
        {"CustomerID": 100.0, "InvoiceNo": "A1", "StockCode": "X1",
         "Quantity": 1, "UnitPrice": 1.0, "Revenue": 1.0,
         "InvoiceDate": pd.Timestamp("2011-01-01"), "Country": "United Kingdom"},
        {"CustomerID": 100.0, "InvoiceNo": "A2", "StockCode": "X1",
         "Quantity": 1, "UnitPrice": 1.0, "Revenue": 1.0,
         "InvoiceDate": pd.Timestamp("2011-03-01"), "Country": "United Kingdom"},
        {"CustomerID": 100.0, "InvoiceNo": "A3", "StockCode": "X1",
         "Quantity": 1, "UnitPrice": 1.0, "Revenue": 1.0,
         "InvoiceDate": pd.Timestamp("2011-05-01"), "Country": "Germany"},
    ]
    df = pd.DataFrame(rows)
    features = compute_customer_features(df, snapshot=SNAPSHOT_DATE)
    assert features.loc[100.0, "Country"] == "United Kingdom"


def test_country_tiebreak_by_most_recent():
    """On a frequency tie, most-recent-purchase country wins."""
    rows = [
        {"CustomerID": 100.0, "InvoiceNo": "A1", "StockCode": "X1",
         "Quantity": 1, "UnitPrice": 1.0, "Revenue": 1.0,
         "InvoiceDate": pd.Timestamp("2011-01-01"), "Country": "United Kingdom"},
        {"CustomerID": 100.0, "InvoiceNo": "A2", "StockCode": "X1",
         "Quantity": 1, "UnitPrice": 1.0, "Revenue": 1.0,
         "InvoiceDate": pd.Timestamp("2011-06-01"), "Country": "Germany"},
    ]
    df = pd.DataFrame(rows)
    features = compute_customer_features(df, snapshot=SNAPSHOT_DATE)
    # Both countries appear once; Germany is more recent → Germany wins
    assert features.loc[100.0, "Country"] == "Germany"


# ---------------------------------------------------------------------------
# RFM scoring
# ---------------------------------------------------------------------------


def test_rfm_score_range():
    """RFM_Score must be in [3, 15] for all customers."""
    df = _make_obs_large()
    features = compute_customer_features(df, snapshot=SNAPSHOT_DATE)
    scored = add_rfm_scores(features)
    assert scored["RFM_Score"].between(3, 15).all()


def test_rfm_segment_labels():
    """RFM_Segment must be one of the four defined labels."""
    valid_labels = {"Very High Value", "High Value", "Medium Value", "Lower Value"}
    df = _make_obs_large()
    features = compute_customer_features(df, snapshot=SNAPSHOT_DATE)
    scored = add_rfm_scores(features)
    assert set(scored["RFM_Segment"].unique()).issubset(valid_labels)


def test_rfm_score_range_small_population():
    """RFM scoring must not raise errors on a small 2-customer population."""
    df = _make_obs()
    features = compute_customer_features(df, snapshot=SNAPSHOT_DATE)
    scored = add_rfm_scores(features)
    assert scored["RFM_Score"].between(3, 15).all()


def test_rfm_columns_named_r_f_m():
    """Output columns must be R, F, M — not R_Score, F_Score, M_Score."""
    df = _make_obs_large()
    features = compute_customer_features(df, snapshot=SNAPSHOT_DATE)
    scored = add_rfm_scores(features)
    assert "R" in scored.columns
    assert "F" in scored.columns
    assert "M" in scored.columns
    assert "R_Score" not in scored.columns
    assert "F_Score" not in scored.columns
    assert "M_Score" not in scored.columns


def test_rfm_recency_ordering():
    """Customer with lower recency (more recent) must get higher R score."""
    rows = []
    for cid in range(10):
        for i in range(3):
            rows.append({
                "CustomerID": float(cid + 1),
                "InvoiceNo": f"INV_{cid}_{i}",
                "StockCode": "X",
                "Quantity": 1, "UnitPrice": 1.0, "Revenue": 1.0,
                "InvoiceDate": pd.Timestamp("2011-01-01") + pd.Timedelta(days=cid * 10 + i),
                "Country": "UK",
            })
    df = pd.DataFrame(rows)
    features = compute_customer_features(df, snapshot=SNAPSHOT_DATE)
    scored = add_rfm_scores(features)
    # Customer with lowest recency (most recent) should have highest R
    min_recency_cid = scored["Recency"].idxmin()
    max_recency_cid = scored["Recency"].idxmax()
    assert scored.loc[min_recency_cid, "R"] >= scored.loc[max_recency_cid, "R"]


def test_rfm_segment_thresholds():
    """Verify segment assignment matches documented thresholds."""
    from src.features.rfm import _rfm_label
    assert _rfm_label(15) == "Very High Value"
    assert _rfm_label(13) == "Very High Value"
    assert _rfm_label(12) == "High Value"
    assert _rfm_label(10) == "High Value"
    assert _rfm_label(9) == "Medium Value"
    assert _rfm_label(7) == "Medium Value"
    assert _rfm_label(6) == "Lower Value"
    assert _rfm_label(3) == "Lower Value"


def test_no_missing_values_in_features():
    """Feature table must have no NaN values for the standard feature columns."""
    df = _make_obs()
    features = compute_customer_features(df, snapshot=SNAPSHOT_DATE)
    numeric_cols = [
        "Recency", "Frequency", "Monetary", "AvgOrderValue",
        "UniqueProducts", "TotalItems", "ActiveMonths",
        "TenureDays", "PurchaseSpanDays",
    ]
    assert features[numeric_cols].isna().sum().sum() == 0


# ---------------------------------------------------------------------------
# Frequency quartile grouping (mirrors app/pages/churn_analytics.py logic)
# ---------------------------------------------------------------------------


def _freq_quartile_groups(frequency_series: "pd.Series") -> "pd.Series":
    """Replicate the rank-based quartile logic used in churn_analytics.py."""
    freq_rank = frequency_series.rank(method="first", ascending=True)
    return pd.cut(
        freq_rank,
        bins=4,
        labels=["Q1", "Q2", "Q3", "Q4"],
        include_lowest=True,
    )


def test_freq_quartile_exactly_four_groups():
    """rank+cut must produce exactly Q1, Q2, Q3, Q4 even when Frequency has many ties."""
    # Simulate a heavily tie-heavy discrete Frequency column (like real retail data)
    freq = pd.Series(
        [2] * 10 + [3] * 10 + [4] * 10 + [5] * 10 + [6] * 10
        + [7] * 5 + [8] * 5 + [10] * 5 + [15] * 5,
        name="Frequency",
    )
    groups = _freq_quartile_groups(freq)
    assert set(groups.dropna().unique()) == {"Q1", "Q2", "Q3", "Q4"}, (
        f"Expected exactly Q1–Q4; got {set(groups.dropna().unique())}"
    )


def test_freq_quartile_all_customers_assigned():
    """Every customer must be assigned to exactly one quartile group (no NaN)."""
    freq = pd.Series(
        [2] * 20 + [3] * 15 + [5] * 15 + [10] * 10,
        name="Frequency",
    )
    groups = _freq_quartile_groups(freq)
    assert groups.isna().sum() == 0, (
        f"{groups.isna().sum()} customers have no group assigned"
    )
    assert len(groups) == len(freq)


def test_freq_quartile_no_duplicates_across_groups():
    """No customer index should appear in more than one group."""
    freq = pd.Series(
        [2, 2, 2, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20],
        name="Frequency",
    )
    groups = _freq_quartile_groups(freq)
    # Each position is in exactly one group — guaranteed by pd.cut on unique ranks
    assert len(groups) == len(freq)
    assert groups.isna().sum() == 0


def test_freq_quartile_group_sizes_approximately_equal():
    """Groups should be approximately equal in size (within 2× of each other)."""
    import numpy as np
    rng = np.random.default_rng(42)
    # 1 765 customers, frequency drawn from a realistic discrete distribution
    freq = pd.Series(
        rng.choice(range(2, 50), size=1765, replace=True),
        name="Frequency",
    )
    groups = _freq_quartile_groups(freq)
    counts = groups.value_counts()
    assert counts.min() > 0
    # Each group should hold at least 15% of customers (1765/4 ≈ 441)
    assert counts.min() >= int(0.15 * len(freq)), (
        f"Smallest group has only {counts.min()} customers: {counts.to_dict()}"
    )


def test_freq_quartile_robust_all_same_frequency():
    """When all customers share the same frequency, rank still assigns 4 groups."""
    freq = pd.Series([5] * 100, name="Frequency")
    groups = _freq_quartile_groups(freq)
    assert set(groups.dropna().unique()) == {"Q1", "Q2", "Q3", "Q4"}
    assert groups.isna().sum() == 0
