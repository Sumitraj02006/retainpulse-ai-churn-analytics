"""
tests/features/test_churn_label.py
------------------------------------
Tests for src/features/churn_label.py — Churn_90D generation,
target integrity, and the leakage guard.
"""

import pandas as pd
import pytest

from src.features.churn_label import generate_churn_label
from src.models.train import NUMERIC_FEATURES
from src.config import FUTURE_START, FUTURE_END, SNAPSHOT_DATE


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


def _make_all_qualified():
    """Qualified purchases spanning observation and future window for 3 customers."""
    rows = [
        # Customer 100 — purchases in future window → Churn_90D = 0
        {"CustomerID": 100.0, "InvoiceNo": "A1",
         "InvoiceDate": pd.Timestamp("2011-05-01"), "Revenue": 10.0},
        {"CustomerID": 100.0, "InvoiceNo": "A2",
         "InvoiceDate": pd.Timestamp("2011-09-15"), "Revenue": 20.0},  # future
        # Customer 200 — no future purchase → Churn_90D = 1
        {"CustomerID": 200.0, "InvoiceNo": "B1",
         "InvoiceDate": pd.Timestamp("2011-03-01"), "Revenue": 5.0},
        # Customer 300 — future purchase right at boundary → Churn_90D = 0
        {"CustomerID": 300.0, "InvoiceNo": "C1",
         "InvoiceDate": pd.Timestamp("2011-11-29"), "Revenue": 15.0},  # end of window
    ]
    return pd.DataFrame(rows)


def _make_obs_and_future():
    """Separate observation and future DataFrames for three customers."""
    obs_rows = [
        {"CustomerID": 10.0, "InvoiceNo": "O1",
         "InvoiceDate": pd.Timestamp("2011-01-01"), "Revenue": 10.0},
        {"CustomerID": 10.0, "InvoiceNo": "O2",
         "InvoiceDate": pd.Timestamp("2011-06-01"), "Revenue": 10.0},
        {"CustomerID": 20.0, "InvoiceNo": "O3",
         "InvoiceDate": pd.Timestamp("2011-02-01"), "Revenue": 5.0},
        {"CustomerID": 20.0, "InvoiceNo": "O4",
         "InvoiceDate": pd.Timestamp("2011-07-01"), "Revenue": 5.0},
    ]
    future_rows = [
        # Customer 10 buys in future → active
        {"CustomerID": 10.0, "InvoiceNo": "F1",
         "InvoiceDate": pd.Timestamp("2011-10-01"), "Revenue": 8.0},
        # Customer 20 has no future purchase → churned
    ]
    df_obs = pd.DataFrame(obs_rows)
    df_future = pd.DataFrame(future_rows)
    return df_obs, df_future


# ---------------------------------------------------------------------------
# Core label tests
# ---------------------------------------------------------------------------


def test_churn_label_active_customer():
    df = _make_all_qualified()
    eligible = [100.0, 200.0, 300.0]
    labels = generate_churn_label(df, eligible)
    assert labels.loc[100.0, "Churn_90D"] == 0


def test_churn_label_churned_customer():
    df = _make_all_qualified()
    eligible = [100.0, 200.0, 300.0]
    labels = generate_churn_label(df, eligible)
    assert labels.loc[200.0, "Churn_90D"] == 1


def test_churn_label_boundary_inclusive():
    """A purchase exactly on FUTURE_END (2011-11-29) counts as active."""
    df = _make_all_qualified()
    eligible = [100.0, 200.0, 300.0]
    labels = generate_churn_label(df, eligible)
    assert labels.loc[300.0, "Churn_90D"] == 0


def test_churn_label_values_are_binary():
    df = _make_all_qualified()
    eligible = [100.0, 200.0, 300.0]
    labels = generate_churn_label(df, eligible)
    assert set(labels["Churn_90D"].unique()).issubset({0, 1})


def test_churn_label_all_eligible_covered():
    df = _make_all_qualified()
    eligible = [100.0, 200.0, 300.0]
    labels = generate_churn_label(df, eligible)
    assert len(labels) == 3


# ---------------------------------------------------------------------------
# Future-window filtering tests
# ---------------------------------------------------------------------------


def test_future_window_start_boundary():
    """A purchase exactly on FUTURE_START (2011-09-01) counts as active."""
    rows = [
        {"CustomerID": 100.0, "InvoiceNo": "A1",
         "InvoiceDate": pd.Timestamp("2011-09-01"), "Revenue": 10.0},
    ]
    df = pd.DataFrame(rows)
    labels = generate_churn_label(df, [100.0])
    assert labels.loc[100.0, "Churn_90D"] == 0


def test_future_window_excludes_before_start():
    """A purchase on 2011-08-31 (SNAPSHOT_DATE) must NOT count towards future window."""
    rows = [
        {"CustomerID": 100.0, "InvoiceNo": "A1",
         "InvoiceDate": pd.Timestamp("2011-08-31"), "Revenue": 10.0},
    ]
    df = pd.DataFrame(rows)
    labels = generate_churn_label(df, [100.0])
    # Purchase is before FUTURE_START → customer has no future purchase → churned
    assert labels.loc[100.0, "Churn_90D"] == 1


def test_future_window_excludes_after_end():
    """A purchase after 2011-11-29 must NOT count towards future window."""
    rows = [
        {"CustomerID": 100.0, "InvoiceNo": "A1",
         "InvoiceDate": pd.Timestamp("2011-11-30"), "Revenue": 10.0},
    ]
    df = pd.DataFrame(rows)
    labels = generate_churn_label(df, [100.0])
    assert labels.loc[100.0, "Churn_90D"] == 1


def test_ineligible_customer_excluded_from_labels():
    """Only eligible customers should appear in the label output."""
    rows = [
        {"CustomerID": 100.0, "InvoiceNo": "A1",
         "InvoiceDate": pd.Timestamp("2011-10-01"), "Revenue": 10.0},
        {"CustomerID": 999.0, "InvoiceNo": "Z1",
         "InvoiceDate": pd.Timestamp("2011-10-15"), "Revenue": 5.0},
    ]
    df = pd.DataFrame(rows)
    # Only 100 is eligible; 999 is ineligible
    labels = generate_churn_label(df, [100.0])
    assert 999.0 not in labels.index
    assert 100.0 in labels.index


def test_no_missing_churn_values():
    """Every eligible customer must have a non-null Churn_90D."""
    df = _make_all_qualified()
    eligible = [100.0, 200.0, 300.0]
    labels = generate_churn_label(df, eligible)
    assert labels["Churn_90D"].isna().sum() == 0


def test_one_row_per_eligible_customer():
    """Labels output must have exactly one row per eligible customer."""
    df = _make_all_qualified()
    eligible = [100.0, 200.0, 300.0]
    labels = generate_churn_label(df, eligible)
    assert labels.index.nunique() == len(eligible)
    assert len(labels) == len(eligible)


# ---------------------------------------------------------------------------
# Leakage guard
# ---------------------------------------------------------------------------


def test_leakage_guard_numeric_features_no_future_fields():
    """
    NUMERIC_FEATURES used in the model must not contain any field name
    that references the future window.  Post-snapshot field names are
    explicitly prohibited.
    """
    forbidden_substrings = [
        "future",
        "sep",
        "oct",
        "nov",
        "post",
        "after",
        "churn",  # Churn_90D must not be a predictor
    ]
    for feat in NUMERIC_FEATURES:
        feat_lower = feat.lower()
        for forbidden in forbidden_substrings:
            assert forbidden not in feat_lower, (
                f"Feature '{feat}' appears to reference post-snapshot "
                f"information (matched '{forbidden}')."
            )


def test_leakage_guard_churn_90d_not_in_features():
    """Churn_90D must not appear in the model feature list."""
    assert "Churn_90D" not in NUMERIC_FEATURES


def test_leakage_guard_snapshot_is_fixed():
    """SNAPSHOT_DATE must equal 2011-08-31 — never derived from data."""
    assert SNAPSHOT_DATE == pd.Timestamp("2011-08-31")


def test_leakage_guard_future_window_correct():
    """Future window boundaries must be exactly as specified."""
    assert FUTURE_START == pd.Timestamp("2011-09-01")
    assert FUTURE_END == pd.Timestamp("2011-11-29")


def test_leakage_guard_customer_features_schema():
    """customer_features.csv must not contain post-snapshot predictor columns."""
    import pathlib
    csv_path = pathlib.Path("data/processed/customer_features.csv")
    if not csv_path.exists():
        pytest.skip("customer_features.csv not yet generated — skipping schema check")

    df = pd.read_csv(csv_path, nrows=0)
    forbidden_columns = {
        "FutureOrderCount", "FutureRevenue", "FutureRecency",
        "FutureFrequency", "FutureProducts", "future_purchase_count",
    }
    present_forbidden = forbidden_columns.intersection(df.columns)
    assert not present_forbidden, (
        f"customer_features.csv contains forbidden post-snapshot columns: {present_forbidden}"
    )


def test_leakage_guard_required_columns_present():
    """customer_features.csv must contain all required schema columns."""
    import pathlib
    csv_path = pathlib.Path("data/processed/customer_features.csv")
    if not csv_path.exists():
        pytest.skip("customer_features.csv not yet generated")

    df = pd.read_csv(csv_path, nrows=0)
    required = {
        "CustomerID", "Recency", "Frequency", "Monetary", "AvgOrderValue",
        "UniqueProducts", "TotalItems", "ActiveMonths", "TenureDays",
        "PurchaseSpanDays", "Country", "R", "F", "M",
        "RFM_Score", "RFM_Segment", "Churn_90D",
    }
    missing = required - set(df.columns)
    assert not missing, f"customer_features.csv is missing required columns: {missing}"
