"""
tests/test_app_helpers.py
--------------------------
Unit tests for Phase 5 dashboard helper functions and module imports.

Covers:
  - app module imports
  - customer data loading (real processed CSV)
  - required columns exist
  - risk-tier assignment
  - retention-priority assignment
  - customer lookup (valid and invalid IDs)
  - model artifact loading
  - evaluation metrics loading
"""
from __future__ import annotations

import importlib
import json
import pathlib

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
CUSTOMER_FEATURES_PATH = REPO_ROOT / "data" / "processed" / "customer_features.csv"
MODEL_ARTIFACT_PATH = REPO_ROOT / "models" / "churn_model.joblib"
EVAL_JSON_PATH = REPO_ROOT / "reports" / "model_evaluation.json"

# ---------------------------------------------------------------------------
# Helper: synthetic customer DataFrame mirroring production schema
# ---------------------------------------------------------------------------

_REQUIRED_FEATURE_COLS = [
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
    "R",
    "F",
    "M",
    "RFM_Score",
    "RFM_Segment",
    "Churn_90D",
]


def _make_synthetic_df(n: int = 20, seed: int = 42) -> pd.DataFrame:
    """Return a minimal synthetic customer DataFrame for unit tests."""
    rng = np.random.default_rng(seed)
    data = {
        "Recency": rng.integers(1, 365, n),
        "Frequency": rng.integers(2, 30, n),
        "Monetary": rng.uniform(100, 5000, n),
        "AvgOrderValue": rng.uniform(10, 200, n),
        "UniqueProducts": rng.integers(1, 100, n),
        "TotalItems": rng.integers(10, 500, n),
        "ActiveMonths": rng.integers(1, 12, n),
        "TenureDays": rng.integers(30, 365, n),
        "PurchaseSpanDays": rng.integers(30, 365, n),
        "Country": rng.choice(["United Kingdom", "Germany", "France"], n),
        "R": rng.integers(1, 5, n),
        "F": rng.integers(1, 5, n),
        "M": rng.integers(1, 5, n),
        "RFM_Score": rng.integers(3, 15, n),
        "RFM_Segment": rng.choice(
            ["Very High Value", "High Value", "Medium Value", "Lower Value"], n
        ),
        "Churn_90D": rng.integers(0, 2, n),
    }
    base_ids = [12347.0, 12348.0, 12352.0, 12356.0, 12358.0, 12359.0, 12360.0,
                12362.0, 12363.0, 12364.0, 12365.0, 12367.0, 12370.0, 12371.0,
                12375.0, 12377.0, 12378.0, 12380.0, 12381.0, 12383.0]
    customer_ids = base_ids[:n]
    return pd.DataFrame(data, index=pd.Index(customer_ids, name="CustomerID"))


# ===========================================================================
# 1. Module import tests
# ===========================================================================


def test_app_utils_imports():
    """app.utils must import without raising."""
    import app.utils  # noqa: F401


def test_app_pages_executive_imports():
    import app.pages.executive  # noqa: F401


def test_app_pages_rfm_analytics_imports():
    import app.pages.rfm_analytics  # noqa: F401


def test_app_pages_churn_analytics_imports():
    import app.pages.churn_analytics  # noqa: F401


def test_app_pages_risk_explorer_imports():
    import app.pages.risk_explorer  # noqa: F401


def test_src_insights_imports():
    import src.insights  # noqa: F401


# ===========================================================================
# 2. Customer data loading
# ===========================================================================


@pytest.mark.skipif(
    not CUSTOMER_FEATURES_PATH.exists(),
    reason="customer_features.csv not present",
)
def test_customer_features_loads():
    df = pd.read_csv(CUSTOMER_FEATURES_PATH, index_col="CustomerID")
    assert len(df) > 0, "customer_features.csv is empty"


@pytest.mark.skipif(
    not CUSTOMER_FEATURES_PATH.exists(),
    reason="customer_features.csv not present",
)
def test_customer_features_required_columns():
    df = pd.read_csv(CUSTOMER_FEATURES_PATH, index_col="CustomerID")
    for col in _REQUIRED_FEATURE_COLS:
        assert col in df.columns, f"Required column '{col}' missing from customer_features.csv"


@pytest.mark.skipif(
    not CUSTOMER_FEATURES_PATH.exists(),
    reason="customer_features.csv not present",
)
def test_customer_features_no_null_customer_id():
    df = pd.read_csv(CUSTOMER_FEATURES_PATH, index_col="CustomerID")
    assert df.index.isnull().sum() == 0, "CustomerID index contains nulls"


# ===========================================================================
# 3. Risk-tier assignment
# ===========================================================================


def test_assign_risk_tiers_valid_labels():
    from src.insights import assign_risk_tiers

    probs = pd.Series([0.1, 0.3, 0.5, 0.7, 0.9])
    tiers = assign_risk_tiers(probs)
    valid = {"High Risk", "Medium Risk", "Low Risk"}
    assert set(tiers.unique()).issubset(valid)


def test_assign_risk_tiers_top_20_percent_high():
    from src.insights import assign_risk_tiers

    probs = pd.Series(np.linspace(0, 1, 100))
    tiers = assign_risk_tiers(probs)
    high_80 = np.percentile(probs, 80)
    for i, p in enumerate(probs):
        if p >= high_80:
            assert tiers.iloc[i] == "High Risk", f"prob={p} should be High Risk"


def test_assign_risk_tiers_bottom_40_percent_low():
    from src.insights import assign_risk_tiers

    probs = pd.Series(np.linspace(0, 1, 100))
    tiers = assign_risk_tiers(probs)
    low_40 = np.percentile(probs, 40)
    for i, p in enumerate(probs):
        if p < low_40:
            assert tiers.iloc[i] == "Low Risk", f"prob={p} should be Low Risk"


def test_assign_risk_tiers_distribution():
    """With uniform distribution over 100 points, tiers should be ~40/40/20."""
    from src.insights import assign_risk_tiers

    probs = pd.Series(np.linspace(0, 1, 100))
    tiers = assign_risk_tiers(probs)
    counts = tiers.value_counts()
    # Top 20% = 20 high, middle 40% = 40 medium, bottom 40% = 40 low
    # Allow ±2 for boundary effects
    assert abs(counts.get("High Risk", 0) - 20) <= 2
    assert abs(counts.get("Medium Risk", 0) - 40) <= 2
    assert abs(counts.get("Low Risk", 0) - 40) <= 2


# ===========================================================================
# 4. Retention priority assignment
# ===========================================================================


def test_retention_priority_high_risk_very_high_value():
    from src.insights import assign_retention_priority

    result = assign_retention_priority("High Risk", "Very High Value")
    assert result == "Priority retention attention"


def test_retention_priority_high_risk_medium_value():
    from src.insights import assign_retention_priority

    result = assign_retention_priority("High Risk", "Medium Value")
    assert result == "Targeted attention"


def test_retention_priority_high_risk_lower_value():
    from src.insights import assign_retention_priority

    result = assign_retention_priority("High Risk", "Lower Value")
    assert result == "Lower-cost intervention"


def test_retention_priority_medium_risk_high_value():
    from src.insights import assign_retention_priority

    result = assign_retention_priority("Medium Risk", "High Value")
    assert result == "Monitor / targeted engagement"


def test_retention_priority_low_risk_high_value():
    from src.insights import assign_retention_priority

    result = assign_retention_priority("Low Risk", "High Value")
    assert result == "Relationship / loyalty attention"


def test_retention_priority_low_risk_lower_value():
    from src.insights import assign_retention_priority

    result = assign_retention_priority("Low Risk", "Lower Value")
    assert result == "Standard engagement"


def test_retention_priority_unknown_combination_defaults():
    from src.insights import assign_retention_priority

    result = assign_retention_priority("Unknown Tier", "Unknown Segment")
    assert result == "Standard engagement"


# ===========================================================================
# 5. Customer lookup
# ===========================================================================


def test_lookup_customer_valid_id():
    from app.utils import lookup_customer

    df = _make_synthetic_df()
    cid = df.index[0]
    result = lookup_customer(df, str(cid))
    assert len(result) == 1
    assert result.index[0] == cid


def test_lookup_customer_invalid_id_returns_empty():
    from app.utils import lookup_customer

    df = _make_synthetic_df()
    result = lookup_customer(df, "99999999")
    assert len(result) == 0


def test_lookup_customer_non_numeric_returns_empty():
    from app.utils import lookup_customer

    df = _make_synthetic_df()
    result = lookup_customer(df, "not_a_number")
    assert len(result) == 0


def test_lookup_customer_empty_string_returns_empty():
    from app.utils import lookup_customer

    df = _make_synthetic_df()
    result = lookup_customer(df, "")
    assert len(result) == 0


def test_lookup_customer_integer_input():
    """Lookup should accept integer as well as string CustomerID."""
    from app.utils import lookup_customer

    df = _make_synthetic_df()
    cid = df.index[2]
    result = lookup_customer(df, int(cid))
    assert len(result) == 1


# ===========================================================================
# 6. Model artifact loading
# ===========================================================================


@pytest.mark.skipif(
    not MODEL_ARTIFACT_PATH.exists(),
    reason="churn_model.joblib not present",
)
def test_model_artifact_loads():
    import warnings
    import joblib

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = joblib.load(MODEL_ARTIFACT_PATH)
    assert model is not None


@pytest.mark.skipif(
    not MODEL_ARTIFACT_PATH.exists(),
    reason="churn_model.joblib not present",
)
def test_model_artifact_has_predict_proba():
    import warnings
    import joblib

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = joblib.load(MODEL_ARTIFACT_PATH)
    assert hasattr(model, "predict_proba")


# ===========================================================================
# 7. Evaluation metrics loading
# ===========================================================================


@pytest.mark.skipif(
    not EVAL_JSON_PATH.exists(),
    reason="model_evaluation.json not present",
)
def test_evaluation_json_loads():
    with open(EVAL_JSON_PATH, encoding="utf-8") as fh:
        data = json.load(fh)
    assert isinstance(data, dict)


@pytest.mark.skipif(
    not EVAL_JSON_PATH.exists(),
    reason="model_evaluation.json not present",
)
def test_evaluation_json_required_keys():
    with open(EVAL_JSON_PATH, encoding="utf-8") as fh:
        data = json.load(fh)
    required = {
        "accuracy", "precision", "recall", "f1",
        "roc_auc", "pr_auc", "default_threshold",
        "threshold_analysis", "feature_coefficients",
    }
    missing = required - data.keys()
    assert not missing, f"Missing keys in evaluation JSON: {missing}"


@pytest.mark.skipif(
    not EVAL_JSON_PATH.exists(),
    reason="model_evaluation.json not present",
)
def test_evaluation_json_metrics_in_range():
    with open(EVAL_JSON_PATH, encoding="utf-8") as fh:
        data = json.load(fh)
    for key in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
        val = data[key]
        assert 0.0 <= val <= 1.0, f"Metric '{key}' = {val} out of [0, 1]"


@pytest.mark.skipif(
    not EVAL_JSON_PATH.exists(),
    reason="model_evaluation.json not present",
)
def test_evaluation_json_default_threshold_is_05():
    with open(EVAL_JSON_PATH, encoding="utf-8") as fh:
        data = json.load(fh)
    assert abs(data["default_threshold"] - 0.50) < 1e-6, (
        f"Default threshold should be 0.50, got {data['default_threshold']}"
    )


# ===========================================================================
# 8. build_customer_risk_table integration
# ===========================================================================


def test_build_customer_risk_table_adds_columns():
    from src.insights import build_customer_risk_table

    df = _make_synthetic_df()
    probs = pd.Series(np.random.default_rng(0).uniform(0, 1, len(df)), index=df.index)
    result = build_customer_risk_table(df, probs)
    assert "Churn_Probability" in result.columns
    assert "Risk_Tier" in result.columns
    assert "Retention_Priority" in result.columns


def test_build_customer_risk_table_no_raw_col():
    """Churn_Probability_Raw must be dropped from the output."""
    from src.insights import build_customer_risk_table

    df = _make_synthetic_df()
    probs = pd.Series(np.random.default_rng(1).uniform(0, 1, len(df)), index=df.index)
    result = build_customer_risk_table(df, probs)
    assert "Churn_Probability_Raw" not in result.columns


def test_build_customer_risk_table_probability_range():
    """Churn_Probability must be 0–100 (percentage)."""
    from src.insights import build_customer_risk_table

    df = _make_synthetic_df()
    probs = pd.Series(np.random.default_rng(2).uniform(0, 1, len(df)), index=df.index)
    result = build_customer_risk_table(df, probs)
    assert result["Churn_Probability"].between(0, 100).all()


# ===========================================================================
# 9. fmt_gbp helper
# ===========================================================================


def test_fmt_gbp_basic():
    from app.utils import fmt_gbp

    assert fmt_gbp(1234.56) == "£1,235"
    assert fmt_gbp(0) == "£0"
    assert fmt_gbp(1000000) == "£1,000,000"


# ===========================================================================
# 10. load_transactions observation-period boundary (regression)
# ===========================================================================


def test_load_transactions_observation_boundary(monkeypatch, tmp_path):
    """All three rows on/before 2011-08-31 are kept; the row after is excluded.

    Regression guard for the calendar-date comparison introduced to replace the
    timestamp comparison ``InvoiceDate <= OBSERVATION_END``, which incorrectly
    dropped transactions occurring later on 2011-08-31 (e.g. 23:59:59).
    """
    import app.utils as utils

    rows = [
        # before the boundary
        {"InvoiceDate": pd.Timestamp("2011-08-30 12:00:00"), "Amount": 1.0},
        # exactly at midnight on boundary day
        {"InvoiceDate": pd.Timestamp("2011-08-31 00:00:00"), "Amount": 2.0},
        # later on boundary day — must be retained
        {"InvoiceDate": pd.Timestamp("2011-08-31 23:59:59"), "Amount": 3.0},
        # one day after boundary — must be excluded
        {"InvoiceDate": pd.Timestamp("2011-09-01 00:00:00"), "Amount": 4.0},
    ]
    df = pd.DataFrame(rows)
    csv_path = tmp_path / "qualified_transactions.csv"
    df.to_csv(csv_path, index=False)

    # Patch the path constant and bypass @st.cache_data
    monkeypatch.setattr(utils, "QUALIFIED_TRANSACTIONS_PATH", csv_path)
    # Clear any cached result so the patched path is used
    try:
        utils.load_transactions.clear()
    except AttributeError:
        pass

    result = utils.load_transactions()
    assert result is not None, "load_transactions() returned None unexpectedly"
    assert len(result) == 3, (
        f"Expected 3 rows (all on/before 2011-08-31), got {len(result)}. "
        f"Returned dates: {result['InvoiceDate'].tolist()}"
    )
    excluded_dates = result[result["InvoiceDate"] >= pd.Timestamp("2011-09-01")]
    assert len(excluded_dates) == 0, (
        "Row(s) after 2011-08-31 were not excluded: "
        f"{excluded_dates['InvoiceDate'].tolist()}"
    )
