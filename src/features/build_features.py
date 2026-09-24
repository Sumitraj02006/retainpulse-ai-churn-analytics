"""
src/features/build_features.py
--------------------------------
Phase 3 pipeline: load qualified_transactions.csv, split observation/future
windows, determine customer eligibility, compute features, score RFM, attach
Churn_90D label, and save customer_features.csv + rfm_churn_summary.json.

Usage (from repo root):
    python -m src.features.build_features
"""

import json
import sys
import pathlib

import pandas as pd

# Allow running as a script from any working directory
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

from src.config import (
    CUSTOMER_FEATURES_PATH,
    FUTURE_END,
    FUTURE_START,
    OBSERVATION_END,
    PROCESSED_DIR,
    SNAPSHOT_DATE,
)
from src.data.clean import get_eligible_customer_ids
from src.data.load import load_qualified_transactions
from src.features.churn_label import generate_churn_label
from src.features.rfm import add_rfm_scores, compute_customer_features


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RFM_CHURN_SUMMARY_PATH = PROCESSED_DIR / "rfm_churn_summary.json"

_CUSTOMER_FEATURES_OUTPUT_COLS = [
    "CustomerID",
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


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def build_customer_features(
    df_qualified: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Run the full Phase 3 feature-engineering and labelling pipeline.

    Parameters
    ----------
    df_qualified : pd.DataFrame, optional
        Pre-loaded qualified transactions.  If None, loads from
        QUALIFIED_TRANSACTIONS_PATH.

    Returns
    -------
    pd.DataFrame
        One row per eligible customer with all feature columns and Churn_90D.
        Index is the default integer range; CustomerID is a plain column.
    """
    # ------------------------------------------------------------------
    # 1. Load source data
    # ------------------------------------------------------------------
    if df_qualified is None:
        df_qualified = load_qualified_transactions()

    # ------------------------------------------------------------------
    # 2. Observation / future splits (date-level to include all intraday
    #    transactions on the snapshot and window boundary dates)
    # ------------------------------------------------------------------
    snap_date = SNAPSHOT_DATE.date()
    future_start_date = FUTURE_START.date()
    future_end_date = FUTURE_END.date()
    _dates = df_qualified["InvoiceDate"].dt.date

    df_obs = df_qualified[_dates <= snap_date].copy()
    df_future = df_qualified[
        (_dates >= future_start_date) & (_dates <= future_end_date)
    ].copy()

    print(f"Observation transactions : {len(df_obs):,}")
    print(f"Future transactions      : {len(df_future):,}")

    # ------------------------------------------------------------------
    # 3. Customer eligibility (observation period only)
    # ------------------------------------------------------------------
    eligible_ids = get_eligible_customer_ids(df_obs, snapshot=SNAPSHOT_DATE)
    print(f"Eligible customers       : {len(eligible_ids):,}")

    # ------------------------------------------------------------------
    # 4. Customer features (observation period, eligible customers only)
    # ------------------------------------------------------------------
    df_eligible_obs = df_obs[df_obs["CustomerID"].isin(eligible_ids)].copy()
    features = compute_customer_features(df_eligible_obs, snapshot=SNAPSHOT_DATE)

    # ------------------------------------------------------------------
    # 5. RFM scoring
    # ------------------------------------------------------------------
    features = add_rfm_scores(features)

    # ------------------------------------------------------------------
    # 6. Churn_90D label (uses full qualified set for future-window lookup)
    # ------------------------------------------------------------------
    labels = generate_churn_label(df_qualified, eligible_ids)
    features = features.join(labels, how="left")

    # ------------------------------------------------------------------
    # 7. Integrity assertions
    # ------------------------------------------------------------------
    assert features["Churn_90D"].isna().sum() == 0, "Missing Churn_90D values"
    assert set(features["Churn_90D"].unique()).issubset(
        {0, 1}
    ), "Churn_90D contains values other than 0 and 1"
    assert features.index.nunique() == len(features), "Duplicate CustomerIDs"

    forbidden_cols = {
        "FutureOrderCount", "FutureRevenue", "FutureRecency",
        "FutureFrequency", "FutureProducts", "future_purchase_count",
    }
    leakage = forbidden_cols.intersection(features.columns)
    assert not leakage, f"Leakage columns found in feature table: {leakage}"

    print(f"Churn_90D=1 (inactive)   : {int(features['Churn_90D'].sum()):,}")
    print(f"Churn_90D=0 (active)     : {int((features['Churn_90D'] == 0).sum()):,}")

    # ------------------------------------------------------------------
    # 8. Reset index so CustomerID is a plain column
    # ------------------------------------------------------------------
    features = features.reset_index()  # CustomerID moves from index → column

    return features


def save_customer_features(
    features: pd.DataFrame,
    output_path: pathlib.Path = CUSTOMER_FEATURES_PATH,
) -> pathlib.Path:
    """Save the customer feature table to CSV.

    Only the canonical output columns are written; no future-data columns.
    """
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cols = [c for c in _CUSTOMER_FEATURES_OUTPUT_COLS if c in features.columns]
    features[cols].to_csv(output_path, index=False)
    print(f"Saved customer features  : {output_path}")
    return output_path


def save_rfm_churn_summary(
    features: pd.DataFrame,
    df_obs: pd.DataFrame,
    df_future: pd.DataFrame,
    output_path: pathlib.Path = RFM_CHURN_SUMMARY_PATH,
) -> pathlib.Path:
    """Compute and save rfm_churn_summary.json.

    Parameters
    ----------
    features : pd.DataFrame
        The customer features table (including RFM scores and Churn_90D).
    df_obs : pd.DataFrame
        Observation-period transactions.
    df_future : pd.DataFrame
        Future-window transactions.
    output_path : pathlib.Path
        Destination for the JSON file.
    """
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    churn_counts = features["Churn_90D"].value_counts().to_dict()
    n_total = len(features)
    n_churned = int(churn_counts.get(1, 0))
    n_active = int(churn_counts.get(0, 0))

    rfm_score_dist = features["RFM_Score"].value_counts().sort_index().to_dict()
    rfm_segment_counts = features["RFM_Segment"].value_counts().to_dict()

    summary = {
        "snapshot_date": str(SNAPSHOT_DATE.date()),
        "observation_end": str(OBSERVATION_END.date()),
        "future_start": str(FUTURE_START.date()),
        "future_end": str(FUTURE_END.date()),
        "observation_transaction_rows": int(len(df_obs)),
        "future_transaction_rows": int(len(df_future)),
        "eligible_customers": int(n_total),
        "rfm_score_distribution": {str(k): int(v) for k, v in rfm_score_dist.items()},
        "rfm_segment_counts": {str(k): int(v) for k, v in rfm_segment_counts.items()},
        "churn_90d": {
            "class_0_count": n_active,
            "class_1_count": n_churned,
            "class_0_pct": round(n_active / n_total * 100, 4) if n_total else 0.0,
            "class_1_pct": round(n_churned / n_total * 100, 4) if n_total else 0.0,
            "future_inactive_count": n_churned,
            "future_active_count": n_active,
        },
        "rfm_feature_summary": {
            col: {
                "min": round(float(features[col].min()), 4),
                "max": round(float(features[col].max()), 4),
                "mean": round(float(features[col].mean()), 4),
                "median": round(float(features[col].median()), 4),
            }
            for col in ["Recency", "Frequency", "Monetary", "AvgOrderValue",
                        "TenureDays", "PurchaseSpanDays", "ActiveMonths",
                        "UniqueProducts", "TotalItems", "RFM_Score"]
        },
    }

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Saved RFM/churn summary  : {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df_qualified = load_qualified_transactions()

    _dates = df_qualified["InvoiceDate"].dt.date
    df_obs = df_qualified[_dates <= SNAPSHOT_DATE.date()].copy()
    df_future = df_qualified[
        (_dates >= FUTURE_START.date()) & (_dates <= FUTURE_END.date())
    ].copy()

    features = build_customer_features(df_qualified)
    save_customer_features(features)
    save_rfm_churn_summary(features, df_obs, df_future)

    print("\nPhase 3 pipeline complete.")
    print(f"  Rows in customer_features.csv : {len(features):,}")
    print(
        f"  Churn_90D=1 rate              : "
        f"{features['Churn_90D'].mean() * 100:.2f}%"
    )
