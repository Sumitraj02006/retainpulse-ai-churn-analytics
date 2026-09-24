"""
app/utils.py
-------------
Shared helpers for the Streamlit dashboard.

Responsibilities:
- Loading and caching the enriched customer risk table (features + model
  probabilities + risk tiers + retention priorities).
- Loading and caching evaluation artifacts (JSON, saved PNGs).
- Loading the qualified-transaction table for trend analysis.
- Customer lookup helpers.
- Risk-tier and retention-priority computation (delegates to src/insights.py).

No model training occurs here.  All probabilities come from the saved
churn_model.joblib applied to customer_features.csv.
"""

from __future__ import annotations

import json
import pathlib
import warnings

import pandas as pd
import streamlit as st

from src.config import (
    CUSTOMER_FEATURES_PATH,
    MODEL_ARTIFACT_PATH,
    QUALIFIED_TRANSACTIONS_PATH,
    REPORTS_DIR,
    OBSERVATION_END,
)
from src.insights import assign_risk_tiers, assign_retention_priority
from src.models.train import NUMERIC_FEATURES

# ---------------------------------------------------------------------------
# Column order for the Risk Explorer display
# ---------------------------------------------------------------------------

EXPLORER_COLS = [
    "Country",
    "Recency",
    "Frequency",
    "Monetary",
    "AvgOrderValue",
    "UniqueProducts",
    "TotalItems",
    "ActiveMonths",
    "TenureDays",
    "PurchaseSpanDays",
    "RFM_Score",
    "RFM_Segment",
    "Churn_Probability",
    "Risk_Tier",
    "Retention_Priority",
]

# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading model…")
def load_model():
    """Load the churn pipeline from disk; return None if absent."""
    if not MODEL_ARTIFACT_PATH.exists():
        return None
    import joblib

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return joblib.load(MODEL_ARTIFACT_PATH)


@st.cache_data(show_spinner="Loading customer data…")
def load_enriched_customers() -> pd.DataFrame:
    """Load customer_features.csv, run the saved model, attach risk tiers.

    Returns a DataFrame indexed by CustomerID with all feature columns plus:
      - Churn_Probability  (float, 0–100 rounded to 2 dp)
      - Churn_Probability_Raw  (float, 0–1)
      - Risk_Tier  (str)
      - Retention_Priority  (str)

    Raises st.stop() with a user-facing message if any required file is absent.
    """
    if not CUSTOMER_FEATURES_PATH.exists():
        st.error(
            f"Customer feature file not found at `{CUSTOMER_FEATURES_PATH}`. "
            "Please run notebooks 01–04 to generate the processed dataset."
        )
        st.stop()

    df = pd.read_csv(CUSTOMER_FEATURES_PATH, index_col="CustomerID")

    model = load_model()
    if model is None:
        st.error(
            f"Model artifact not found at `{MODEL_ARTIFACT_PATH}`. "
            "Please run notebook 04 to train the model."
        )
        st.stop()

    missing_feats = [c for c in NUMERIC_FEATURES if c not in df.columns]
    if missing_feats:
        st.error(
            f"Required predictor columns are missing from customer_features.csv: "
            f"{missing_feats}"
        )
        st.stop()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        probs_raw = model.predict_proba(df[NUMERIC_FEATURES])[:, 1]

    df["Churn_Probability_Raw"] = probs_raw
    df["Churn_Probability"] = (probs_raw * 100).round(2)
    df["Risk_Tier"] = assign_risk_tiers(pd.Series(probs_raw, index=df.index))

    if "RFM_Segment" in df.columns:
        df["Retention_Priority"] = df.apply(
            lambda row: assign_retention_priority(row["Risk_Tier"], row["RFM_Segment"]),
            axis=1,
        )
    else:
        df["Retention_Priority"] = "Standard engagement"

    return df


@st.cache_data(show_spinner=False)
def load_evaluation_json() -> dict | None:
    """Load reports/model_evaluation.json; return None if absent."""
    path = REPORTS_DIR / "model_evaluation.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


@st.cache_data(show_spinner=False)
def load_transactions() -> pd.DataFrame | None:
    """Load qualified_transactions.csv for trend charts; return None if absent."""
    if not QUALIFIED_TRANSACTIONS_PATH.exists():
        return None
    tx = pd.read_csv(QUALIFIED_TRANSACTIONS_PATH, parse_dates=["InvoiceDate"])
    # Restrict to observation period only (no future-window leakage)
    return tx[tx["InvoiceDate"].dt.date <= OBSERVATION_END.date()].copy()


# ---------------------------------------------------------------------------
# Customer lookup
# ---------------------------------------------------------------------------


def lookup_customer(df: pd.DataFrame, customer_id: float | int | str) -> pd.DataFrame:
    """Return a one-row DataFrame for *customer_id*, or empty if not found.

    Accepts string, int, or float representations of the CustomerID.
    """
    try:
        cid = float(customer_id)
    except (ValueError, TypeError):
        return df.iloc[0:0]  # empty, same schema

    if cid in df.index:
        return df.loc[[cid]]
    return df.iloc[0:0]


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def fmt_gbp(value: float) -> str:
    """Format a numeric value as GBP with thousands separator."""
    return f"£{value:,.0f}"


def fmt_pct(value: float, decimals: int = 1) -> str:
    """Format a 0–1 fraction as a percentage string."""
    return f"{value * 100:.{decimals}f}%"


# ---------------------------------------------------------------------------
# Theme & Color Palettes
# ---------------------------------------------------------------------------

RISK_PALETTE = {
    "High Risk": "#EF4444",    # Crimson Red
    "Medium Risk": "#F59E0B",  # Warm Amber
    "Low Risk": "#10B981",     # Emerald Green
}

SEGMENT_PALETTE = {
    "Very High Value": "#8B5CF6",  # Violet
    "High Value": "#3B82F6",       # Blue
    "Medium Value": "#06B6D4",     # Teal/Cyan
    "Lower Value": "#64748B",      # Slate
}


def apply_chart_style(fig, title=""):
    """Apply unified dark mode styling to Plotly figures."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.4)",
        font=dict(family="Inter, system-ui, sans-serif", color="#F8FAFC", size=12),
        margin=dict(t=40 if title else 20, b=30, l=30, r=30),
        title=dict(text=title, font=dict(size=15, color="#F8FAFC")),
        legend=dict(
            bgcolor="rgba(15, 23, 42, 0.6)",
            bordercolor="rgba(255, 255, 255, 0.1)",
            borderwidth=1,
            font=dict(size=11),
        ),
        xaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.07)",
            zerolinecolor="rgba(255, 255, 255, 0.1)",
        ),
        yaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.07)",
            zerolinecolor="rgba(255, 255, 255, 0.1)",
        ),
    )
    return fig


def render_kpi_card(title: str, value: str, subtitle: str = "", badge: str = "", accent: str = "#6366F1", icon: str = "📈"):
    """Render a modern glassmorphic KPI card HTML component."""
    badge_html = f'<span style="background:{accent}22; color:{accent}; border:1px solid {accent}55; padding: 2px 8px; border-radius: 12px; font-size: 0.72rem; font-weight: 600;">{badge}</span>' if badge else ''
    sub_html = f'<div style="color: #94A3B8; font-size: 0.78rem; margin-top: 4px;">{subtitle}</div>' if subtitle else ''
    
    html = f"""
    <div style="
        background: linear-gradient(145deg, rgba(21, 29, 48, 0.85), rgba(15, 23, 42, 0.75));
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-top: 3px solid {accent};
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        margin-bottom: 12px;
    ">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
            <span style="color: #CBD5E1; font-size: 0.85rem; font-weight: 500;">{icon} {title}</span>
            {badge_html}
        </div>
        <div style="color: #FFFFFF; font-size: 1.6rem; font-weight: 700; letter-spacing: -0.5px;">{value}</div>
        {sub_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Evaluation artefact paths
# ---------------------------------------------------------------------------

CONFUSION_MATRIX_PATH = REPORTS_DIR / "confusion_matrix.png"
ROC_CURVE_PATH = REPORTS_DIR / "roc_curve.png"
PR_CURVE_PATH = REPORTS_DIR / "precision_recall_curve.png"

