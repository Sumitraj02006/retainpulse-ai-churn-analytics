"""
tests/models/test_train.py
---------------------------
Tests for src/models/train.py — pipeline construction, training,
probability output, stratified split, leakage guards, and artifact I/O.
"""

import pathlib
import tempfile

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.models.train import (
    NUMERIC_FEATURES,
    build_pipeline,
    load_pipeline,
    save_pipeline,
    split_data,
    train,
)
from src.insights import assign_risk_tiers


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_synthetic_dataset(n=200, random_seed=42):
    """Return synthetic features and churn labels for pipeline testing."""
    rng = np.random.default_rng(random_seed)
    data = {
        "Recency": rng.integers(1, 365, n).astype(float),
        "Frequency": rng.integers(1, 50, n).astype(float),
        "Monetary": rng.uniform(10, 5000, n),
        "AvgOrderValue": rng.uniform(5, 200, n),
        "UniqueProducts": rng.integers(1, 100, n).astype(float),
        "TotalItems": rng.integers(1, 500, n).astype(float),
        "ActiveMonths": rng.integers(1, 12, n).astype(float),
        "TenureDays": rng.integers(30, 365, n).astype(float),
        "PurchaseSpanDays": rng.integers(30, 365, n).astype(float),
    }
    features = pd.DataFrame(data, index=pd.RangeIndex(n, name="CustomerID"))
    labels = pd.DataFrame(
        {"Churn_90D": rng.integers(0, 2, n)},
        index=pd.RangeIndex(n, name="CustomerID"),
    )
    return features, labels


# ---------------------------------------------------------------------------
# Pipeline construction tests
# ---------------------------------------------------------------------------


def test_build_pipeline_returns_pipeline():
    pipe = build_pipeline()
    assert isinstance(pipe, Pipeline)


def test_pipeline_has_scaler_and_clf():
    pipe = build_pipeline()
    assert "scaler" in pipe.named_steps
    assert "clf" in pipe.named_steps


def test_pipeline_clf_class_weight_balanced():
    pipe = build_pipeline()
    assert pipe.named_steps["clf"].class_weight == "balanced"


def test_pipeline_clf_random_state():
    pipe = build_pipeline()
    assert pipe.named_steps["clf"].random_state == 42


# ---------------------------------------------------------------------------
# Training tests
# ---------------------------------------------------------------------------


def test_model_pipeline_trains_without_error():
    features, labels = _make_synthetic_dataset()
    X_train, X_test, y_train, y_test = split_data(features, labels)
    pipeline = train(X_train, y_train)
    assert pipeline is not None


def test_model_probability_output_shape():
    features, labels = _make_synthetic_dataset()
    X_train, X_test, y_train, y_test = split_data(features, labels)
    pipeline = train(X_train, y_train)
    probs = pipeline.predict_proba(X_test)
    assert probs.shape[1] == 2  # binary: [P(0), P(1)]
    assert probs.shape[0] == len(X_test)


def test_model_probabilities_sum_to_one():
    features, labels = _make_synthetic_dataset()
    X_train, X_test, y_train, y_test = split_data(features, labels)
    pipeline = train(X_train, y_train)
    probs = pipeline.predict_proba(X_test)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)


def test_model_binary_predictions():
    """predict() must return only 0 and 1."""
    features, labels = _make_synthetic_dataset()
    X_train, X_test, y_train, y_test = split_data(features, labels)
    pipeline = train(X_train, y_train)
    preds = pipeline.predict(X_test)
    assert set(preds).issubset({0, 1})


# ---------------------------------------------------------------------------
# Stratified split tests
# ---------------------------------------------------------------------------


def test_split_data_train_test_sizes():
    """80/20 stratified split: test size must be 20% of total."""
    features, labels = _make_synthetic_dataset(n=200)
    X_train, X_test, y_train, y_test = split_data(features, labels)
    total = len(X_train) + len(X_test)
    assert len(X_test) == pytest.approx(total * 0.2, abs=2)


def test_split_data_stratification():
    """Churn rate in train and test should be approximately equal."""
    features, labels = _make_synthetic_dataset(n=500, random_seed=0)
    X_train, X_test, y_train, y_test = split_data(features, labels)
    assert abs(y_train.mean() - y_test.mean()) < 0.05


def test_split_no_customer_overlap():
    """No CustomerID should appear in both train and test."""
    features, labels = _make_synthetic_dataset()
    X_train, X_test, y_train, y_test = split_data(features, labels)
    overlap = set(X_train.index) & set(X_test.index)
    assert len(overlap) == 0


# ---------------------------------------------------------------------------
# Leakage guard tests
# ---------------------------------------------------------------------------


def test_churn_90d_not_in_numeric_features():
    """Churn_90D must never appear as a predictor."""
    assert "Churn_90D" not in NUMERIC_FEATURES


def test_customer_id_not_in_numeric_features():
    """CustomerID must never appear as a predictor."""
    assert "CustomerID" not in NUMERIC_FEATURES


def test_rfm_scores_not_in_numeric_features():
    """Descriptive RFM score columns must not be used as predictors."""
    forbidden = {"R", "F", "M", "RFM_Score", "RFM_Segment"}
    in_features = forbidden & set(NUMERIC_FEATURES)
    assert not in_features, f"Forbidden predictor(s) in NUMERIC_FEATURES: {in_features}"


def test_split_data_excludes_churn_from_X():
    """split_data must not include Churn_90D in the returned X DataFrames."""
    features, labels = _make_synthetic_dataset()
    X_train, X_test, y_train, y_test = split_data(features, labels)
    assert "Churn_90D" not in X_train.columns
    assert "Churn_90D" not in X_test.columns


def test_feature_names_match_numeric_features():
    """X columns from split_data must match NUMERIC_FEATURES exactly."""
    features, labels = _make_synthetic_dataset()
    X_train, X_test, y_train, y_test = split_data(features, labels)
    assert list(X_train.columns) == NUMERIC_FEATURES
    assert list(X_test.columns) == NUMERIC_FEATURES


# ---------------------------------------------------------------------------
# Model artifact save/load tests
# ---------------------------------------------------------------------------


def test_save_and_load_pipeline():
    """Saved pipeline must load and produce identical probabilities."""
    features, labels = _make_synthetic_dataset()
    X_train, X_test, y_train, y_test = split_data(features, labels)
    pipeline = train(X_train, y_train)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = pathlib.Path(tmpdir) / "model.joblib"
        save_pipeline(pipeline, path=path)
        assert path.exists()

        loaded = load_pipeline(path=path)
        orig_probs = pipeline.predict_proba(X_test)[:, 1]
        loaded_probs = loaded.predict_proba(X_test)[:, 1]
        np.testing.assert_allclose(orig_probs, loaded_probs, rtol=1e-5)


def test_load_pipeline_raises_if_missing():
    with pytest.raises(FileNotFoundError):
        load_pipeline(path=pathlib.Path("/nonexistent/path/model.joblib"))


# ---------------------------------------------------------------------------
# Risk-tier tests (src/insights.py)
# ---------------------------------------------------------------------------


def test_risk_tier_assignment_values():
    probs = pd.Series([0.1, 0.3, 0.5, 0.7, 0.9])
    tiers = assign_risk_tiers(probs)
    valid = {"High Risk", "Medium Risk", "Low Risk"}
    assert set(tiers.unique()).issubset(valid)


def test_risk_tier_top_20_percent_high():
    """Top 20% of probabilities must map to High Risk."""
    probs = pd.Series(np.linspace(0, 1, 100))
    tiers = assign_risk_tiers(probs)
    high_probs = probs[probs >= np.percentile(probs, 80)]
    assert (tiers[high_probs.index] == "High Risk").all()


def test_risk_tier_bottom_40_percent_low():
    """Bottom 40% of probabilities must map to Low Risk."""
    probs = pd.Series(np.linspace(0, 1, 100))
    tiers = assign_risk_tiers(probs)
    low_probs = probs[probs < np.percentile(probs, 40)]
    assert (tiers[low_probs.index] == "Low Risk").all()
