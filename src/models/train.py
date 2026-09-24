"""
src/models/train.py
--------------------
Build, fit, and persist the Logistic Regression churn-prediction pipeline.

The pipeline encapsulates all preprocessing so that the saved artifact is
self-contained and ready for inference without external transformation state.
"""

import pathlib

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import MODEL_ARTIFACT_PATH, MODELS_DIR

# ---------------------------------------------------------------------------
# Feature columns used by the primary model
# ---------------------------------------------------------------------------

NUMERIC_FEATURES = [
    "Recency",
    "Frequency",
    "Monetary",
    "AvgOrderValue",
    "UniqueProducts",
    "TotalItems",
    "ActiveMonths",
    "TenureDays",
    "PurchaseSpanDays",
]


def build_pipeline() -> Pipeline:
    """Return an un-fitted Logistic Regression pipeline."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )


def split_data(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Stratified train/test split on eligible customers.

    Parameters
    ----------
    features : pd.DataFrame
        Customer feature table (indexed by CustomerID).
    labels : pd.DataFrame
        Churn_90D labels (indexed by CustomerID).

    Returns
    -------
    X_train, X_test, y_train, y_test : pd.DataFrame / pd.Series
    """
    combined = features[NUMERIC_FEATURES].join(labels["Churn_90D"], how="inner")
    combined = combined.dropna()

    X = combined[NUMERIC_FEATURES]
    y = combined["Churn_90D"]

    return train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )


def train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    pipeline: Pipeline = None,
) -> Pipeline:
    """Fit the pipeline on training data and return the fitted pipeline."""
    if pipeline is None:
        pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    return pipeline


def save_pipeline(pipeline: Pipeline, path=None) -> pathlib.Path:
    """Persist the fitted pipeline with joblib.

    Parameters
    ----------
    path : path-like, optional
        Override for MODEL_ARTIFACT_PATH.

    Returns
    -------
    pathlib.Path
        Path the artifact was written to.
    """
    path = pathlib.Path(path) if path else MODEL_ARTIFACT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    return path


def load_pipeline(path=None) -> Pipeline:
    """Load a persisted pipeline artifact.

    Raises
    ------
    FileNotFoundError
        If the artifact does not exist at the given path.
    """
    path = pathlib.Path(path) if path else MODEL_ARTIFACT_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at '{path}'. "
            "Run the training notebook (04_churn_model.ipynb) first."
        )
    return joblib.load(path)
