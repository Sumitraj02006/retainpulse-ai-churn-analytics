"""
src/models/evaluate.py
-----------------------
Required metrics, evaluation plots, threshold analysis, coefficient
interpretation, and JSON/CSV artifact saving for the churn pipeline.
"""

import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

from src.models.train import NUMERIC_FEATURES


def compute_metrics(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Return all required evaluation metrics as a dict."""
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "pr_auc": average_precision_score(y_test, y_prob),
    }


def threshold_analysis(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    thresholds: list = None,
) -> pd.DataFrame:
    """Compute Precision, Recall, F1 across classification thresholds.

    Parameters
    ----------
    thresholds : list of float, optional
        Thresholds to evaluate.  Defaults to
        [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60].

    Returns
    -------
    pd.DataFrame
        Columns: threshold, precision, recall, f1
    """
    if thresholds is None:
        thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

    y_prob = pipeline.predict_proba(X_test)[:, 1]
    rows = []
    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        rows.append(
            {
                "threshold": t,
                "precision": precision_score(y_test, y_pred_t, zero_division=0),
                "recall": recall_score(y_test, y_pred_t, zero_division=0),
                "f1": f1_score(y_test, y_pred_t, zero_division=0),
            }
        )
    return pd.DataFrame(rows)


def plot_confusion_matrix(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save_path: pathlib.Path = None,
) -> plt.Figure:
    """Plot and optionally save the confusion matrix."""
    y_pred = pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Active (0)", "Inactive (1)"]
    )
    disp.plot(ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_roc_curve(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save_path: pathlib.Path = None,
) -> plt.Figure:
    """Plot ROC curve and return the figure."""
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_precision_recall_curve(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save_path: pathlib.Path = None,
) -> plt.Figure:
    """Plot Precision-Recall curve and return the figure."""
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision, label=f"PR-AUC = {pr_auc:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def get_feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    """Return a DataFrame of feature coefficients from the fitted LR model.

    The scaler is included in the pipeline; coefficients reflect
    standardised inputs — larger absolute values indicate stronger
    associations with predicted inactivity risk.

    Returns
    -------
    pd.DataFrame
        Columns: feature, coefficient, abs_coefficient
        Sorted by abs_coefficient descending.
    """
    clf = pipeline.named_steps["clf"]
    coefs = clf.coef_[0]
    df = pd.DataFrame({"feature": NUMERIC_FEATURES, "coefficient": coefs})
    df["abs_coefficient"] = df["coefficient"].abs()
    return df.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)


def save_evaluation_json(
    metrics: dict,
    thresh_df: pd.DataFrame,
    coef_df: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    save_path: pathlib.Path,
    default_threshold: float = 0.50,
) -> None:
    """Persist a machine-readable evaluation report as JSON.

    Parameters
    ----------
    metrics : dict
        Output of compute_metrics().
    thresh_df : pd.DataFrame
        Output of threshold_analysis().
    coef_df : pd.DataFrame
        Output of get_feature_importance().
    y_train, y_test : pd.Series
        Training and test labels (for size/distribution records).
    save_path : pathlib.Path
        Destination file path.
    default_threshold : float
        The default classification threshold used.
    """
    report = {
        "train_size": int(len(y_train)),
        "test_size": int(len(y_test)),
        "class_counts": {
            "train": y_train.value_counts().sort_index().to_dict(),
            "test": y_test.value_counts().sort_index().to_dict(),
        },
        "accuracy": round(metrics["accuracy"], 6),
        "precision": round(metrics["precision"], 6),
        "recall": round(metrics["recall"], 6),
        "f1": round(metrics["f1"], 6),
        "roc_auc": round(metrics["roc_auc"], 6),
        "pr_auc": round(metrics["pr_auc"], 6),
        "default_threshold": default_threshold,
        "threshold_analysis": thresh_df.round(6).to_dict(orient="records"),
        "feature_coefficients": coef_df.round(6).to_dict(orient="records"),
    }
    # Convert int64 keys to plain int for JSON serialisation
    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        raise TypeError(f"Not serialisable: {type(obj)}")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=_convert)
