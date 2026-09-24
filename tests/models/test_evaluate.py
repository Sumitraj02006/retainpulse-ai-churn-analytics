"""
tests/models/test_evaluate.py
------------------------------
Tests for src/models/evaluate.py — metrics, threshold analysis,
coefficient extraction, and evaluation JSON saving.
"""

import json
import pathlib
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.models.train import NUMERIC_FEATURES, split_data, train
from src.models.evaluate import (
    compute_metrics,
    get_feature_importance,
    save_evaluation_json,
    threshold_analysis,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_fitted_pipeline(n=200, random_seed=42):
    """Return a fitted pipeline with train/test splits for evaluate tests."""
    rng = np.random.default_rng(random_seed)
    data = {feat: rng.uniform(0, 100, n) for feat in NUMERIC_FEATURES}
    features = pd.DataFrame(data, index=pd.RangeIndex(n, name="CustomerID"))
    labels = pd.DataFrame(
        {"Churn_90D": rng.integers(0, 2, n)},
        index=pd.RangeIndex(n, name="CustomerID"),
    )
    X_train, X_test, y_train, y_test = split_data(features, labels)
    pipeline = train(X_train, y_train)
    return pipeline, X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# compute_metrics tests
# ---------------------------------------------------------------------------


def test_compute_metrics_returns_required_keys():
    pipeline, _, X_test, _, y_test = _make_fitted_pipeline()
    metrics = compute_metrics(pipeline, X_test, y_test)
    required = {"accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"}
    assert required.issubset(metrics.keys())


def test_compute_metrics_values_in_range():
    pipeline, _, X_test, _, y_test = _make_fitted_pipeline()
    metrics = compute_metrics(pipeline, X_test, y_test)
    for k, v in metrics.items():
        assert 0.0 <= v <= 1.0, f"Metric {k} = {v} is out of [0, 1]"


# ---------------------------------------------------------------------------
# threshold_analysis tests
# ---------------------------------------------------------------------------


def test_threshold_analysis_returns_dataframe():
    pipeline, _, X_test, _, y_test = _make_fitted_pipeline()
    result = threshold_analysis(pipeline, X_test, y_test)
    assert isinstance(result, pd.DataFrame)


def test_threshold_analysis_columns():
    pipeline, _, X_test, _, y_test = _make_fitted_pipeline()
    result = threshold_analysis(pipeline, X_test, y_test)
    assert set(result.columns) == {"threshold", "precision", "recall", "f1"}


def test_threshold_analysis_default_thresholds():
    pipeline, _, X_test, _, y_test = _make_fitted_pipeline()
    result = threshold_analysis(pipeline, X_test, y_test)
    expected = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    assert list(result["threshold"]) == expected


def test_threshold_analysis_custom_thresholds():
    pipeline, _, X_test, _, y_test = _make_fitted_pipeline()
    result = threshold_analysis(pipeline, X_test, y_test, thresholds=[0.3, 0.5, 0.7])
    assert list(result["threshold"]) == [0.3, 0.5, 0.7]
    assert len(result) == 3


def test_threshold_analysis_recall_monotone():
    """Higher thresholds should generally produce lower or equal recall."""
    pipeline, _, X_test, _, y_test = _make_fitted_pipeline()
    result = threshold_analysis(pipeline, X_test, y_test)
    # Recall must be non-increasing as threshold increases (monotone by definition)
    recalls = result["recall"].values
    assert all(recalls[i] >= recalls[i + 1] - 1e-6 for i in range(len(recalls) - 1))


# ---------------------------------------------------------------------------
# get_feature_importance tests
# ---------------------------------------------------------------------------


def test_feature_importance_returns_dataframe():
    pipeline, _, X_test, _, y_test = _make_fitted_pipeline()
    coef_df = get_feature_importance(pipeline)
    assert isinstance(coef_df, pd.DataFrame)


def test_feature_importance_columns():
    pipeline, _, _, _, _ = _make_fitted_pipeline()
    coef_df = get_feature_importance(pipeline)
    assert set(coef_df.columns) == {"feature", "coefficient", "abs_coefficient"}


def test_feature_importance_feature_names():
    """Feature names in coefficient table must match NUMERIC_FEATURES."""
    pipeline, _, _, _, _ = _make_fitted_pipeline()
    coef_df = get_feature_importance(pipeline)
    assert set(coef_df["feature"]) == set(NUMERIC_FEATURES)


def test_feature_importance_sorted_descending():
    pipeline, _, _, _, _ = _make_fitted_pipeline()
    coef_df = get_feature_importance(pipeline)
    abs_coefs = coef_df["abs_coefficient"].values
    assert all(abs_coefs[i] >= abs_coefs[i + 1] for i in range(len(abs_coefs) - 1))


# ---------------------------------------------------------------------------
# save_evaluation_json tests
# ---------------------------------------------------------------------------


def test_save_evaluation_json_creates_file():
    pipeline, X_train, X_test, y_train, y_test = _make_fitted_pipeline()
    metrics = compute_metrics(pipeline, X_test, y_test)
    thresh_df = threshold_analysis(pipeline, X_test, y_test)
    coef_df = get_feature_importance(pipeline)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = pathlib.Path(tmpdir) / "eval.json"
        save_evaluation_json(metrics, thresh_df, coef_df, y_train, y_test, path)
        assert path.exists()


def test_save_evaluation_json_required_keys():
    pipeline, X_train, X_test, y_train, y_test = _make_fitted_pipeline()
    metrics = compute_metrics(pipeline, X_test, y_test)
    thresh_df = threshold_analysis(pipeline, X_test, y_test)
    coef_df = get_feature_importance(pipeline)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = pathlib.Path(tmpdir) / "eval.json"
        save_evaluation_json(metrics, thresh_df, coef_df, y_train, y_test, path)
        with open(path) as f:
            data = json.load(f)
        required_keys = {
            "train_size", "test_size", "class_counts",
            "accuracy", "precision", "recall", "f1",
            "roc_auc", "pr_auc", "default_threshold",
            "threshold_analysis", "feature_coefficients",
        }
        assert required_keys.issubset(data.keys())


def test_save_evaluation_json_sizes_correct():
    pipeline, X_train, X_test, y_train, y_test = _make_fitted_pipeline()
    metrics = compute_metrics(pipeline, X_test, y_test)
    thresh_df = threshold_analysis(pipeline, X_test, y_test)
    coef_df = get_feature_importance(pipeline)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = pathlib.Path(tmpdir) / "eval.json"
        save_evaluation_json(metrics, thresh_df, coef_df, y_train, y_test, path)
        with open(path) as f:
            data = json.load(f)
        assert data["train_size"] == len(y_train)
        assert data["test_size"] == len(y_test)
