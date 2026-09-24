"""
app/pages/churn_analytics.py — Page 3: Churn / Risk Analytics

Displays:
  - Churn_90D class distribution
  - Observed inactivity rate by Recency band, Frequency group, Customer-value group
  - Churn probability distribution (from model)
  - Validated Phase 4 model metrics (from evaluation JSON)
  - Confusion matrix, ROC curve, PR curve (saved PNGs)
  - Threshold analysis table/chart
  - Model coefficient associations
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.utils import (
    CONFUSION_MATRIX_PATH,
    PR_CURVE_PATH,
    ROC_CURVE_PATH,
    load_evaluation_json,
    RISK_PALETTE,
    apply_chart_style,
    render_kpi_card,
)
from src.insights import churn_rate_by_group


def render(df: pd.DataFrame, model) -> None:
    if df is None or len(df) == 0:
        st.warning("No customer data available.")
        return

    eval_json = load_evaluation_json()

    # -----------------------------------------------------------------------
    # Section 1 — Observed churn patterns
    # -----------------------------------------------------------------------
    st.markdown("### 🔍 Observed Inactivity (Churn_90D) & Model Output")

    c1, c2 = st.columns(2)

    with c1:
        if "Churn_90D" in df.columns:
            counts = df["Churn_90D"].value_counts().reset_index()
            counts.columns = ["Class", "Count"]
            counts["Label"] = counts["Class"].map({0: "Active (0)", 1: "Inactive (1)"})
            fig_dist = px.bar(
                counts,
                x="Label",
                y="Count",
                color="Label",
                color_discrete_map={"Active (0)": "#10B981", "Inactive (1)": "#EF4444"},
                text="Count",
            )
            fig_dist.update_traces(textposition="outside", texttemplate="%{text:,}")
            fig_dist.update_layout(showlegend=False, xaxis_title="")
            apply_chart_style(fig_dist, "Churn_90D Class Distribution")
            st.plotly_chart(fig_dist, use_container_width=True)
        else:
            st.info("Churn_90D column not available.")

    with c2:
        if "Churn_Probability" in df.columns:
            fig_prob = px.histogram(
                df,
                x="Churn_Probability",
                nbins=40,
                labels={"Churn_Probability": "Predicted Inactivity Probability (%)"},
            )
            fig_prob.update_traces(marker_color="#8B5CF6", marker_line_color="#6D28D9", marker_line_width=1)
            apply_chart_style(fig_prob, "Predicted Churn Probability Distribution (%)")
            st.plotly_chart(fig_prob, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Section 2 — Inactivity rates by behavioural groups
    # -----------------------------------------------------------------------
    st.markdown("### 📊 Inactivity Rates Across Cohorts")

    if "Churn_90D" in df.columns:
        df["Recency_Band"] = pd.cut(
            df["Recency"],
            bins=[0, 30, 60, 90, 120, 180, 9999],
            labels=["≤30d", "31–60d", "61–90d", "91–120d", "121–180d", ">180d"],
        )

        freq_rank = df["Frequency"].rank(method="first", ascending=True)
        df["Frequency_Group"] = pd.cut(
            freq_rank,
            bins=4,
            labels=["Q1 (Lowest)", "Q2", "Q3", "Q4 (Highest)"],
            include_lowest=True,
        )

        df["Value_Group"] = pd.qcut(
            df["Monetary"],
            q=4,
            labels=["Bottom 25%", "25–50%", "50–75%", "Top 25%"],
            duplicates="drop",
        )

        g1, g2, g3 = st.columns(3)

        with g1:
            rate_rec = churn_rate_by_group(df, "Recency_Band")
            fig_r = px.bar(
                rate_rec,
                x="Recency_Band",
                y="churn_rate",
                labels={"Recency_Band": "Recency Band", "churn_rate": "Inactivity Rate"},
                text=rate_rec["churn_rate"].map(lambda v: f"{v:.1%}"),
            )
            fig_r.update_traces(marker_color="#EF4444", textposition="outside")
            fig_r.update_layout(yaxis_tickformat=".0%", yaxis_range=[0, 1], xaxis_title="")
            apply_chart_style(fig_r, "Inactivity by Recency Band")
            st.plotly_chart(fig_r, use_container_width=True)

        with g2:
            rate_freq = churn_rate_by_group(df, "Frequency_Group")
            fig_f = px.bar(
                rate_freq,
                x="Frequency_Group",
                y="churn_rate",
                labels={"Frequency_Group": "Frequency Quartile", "churn_rate": "Inactivity Rate"},
                text=rate_freq["churn_rate"].map(lambda v: f"{v:.1%}"),
            )
            fig_f.update_traces(marker_color="#F59E0B", textposition="outside")
            fig_f.update_layout(yaxis_tickformat=".0%", yaxis_range=[0, 1], xaxis_title="")
            apply_chart_style(fig_f, "Inactivity by Frequency Quartile")
            st.plotly_chart(fig_f, use_container_width=True)

        with g3:
            rate_val = churn_rate_by_group(df, "Value_Group")
            fig_v = px.bar(
                rate_val,
                x="Value_Group",
                y="churn_rate",
                labels={"Value_Group": "Value Group", "churn_rate": "Inactivity Rate"},
                text=rate_val["churn_rate"].map(lambda v: f"{v:.1%}"),
            )
            fig_v.update_traces(marker_color="#3B82F6", textposition="outside")
            fig_v.update_layout(yaxis_tickformat=".0%", yaxis_range=[0, 1], xaxis_title="")
            apply_chart_style(fig_v, "Inactivity by Customer Value")
            st.plotly_chart(fig_v, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Section 3 — Model evaluation metrics (Phase 4)
    # -----------------------------------------------------------------------
    st.markdown("### 🏆 Model Performance Metrics (Logistic Regression)")

    if eval_json is not None:
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        with m1:
            render_kpi_card("Accuracy", f"{eval_json.get('accuracy', 0):.4f}", badge="Eval", accent="#10B981", icon="🎯")
        with m2:
            render_kpi_card("Precision", f"{eval_json.get('precision', 0):.4f}", badge="Eval", accent="#3B82F6", icon="🎯")
        with m3:
            render_kpi_card("Recall", f"{eval_json.get('recall', 0):.4f}", badge="Eval", accent="#F59E0B", icon="🎯")
        with m4:
            render_kpi_card("F1 Score", f"{eval_json.get('f1', 0):.4f}", badge="Eval", accent="#8B5CF6", icon="🎯")
        with m5:
            render_kpi_card("ROC-AUC", f"{eval_json.get('roc_auc', 0):.4f}", badge="Eval", accent="#EC4899", icon="📈")
        with m6:
            render_kpi_card("PR-AUC", f"{eval_json.get('pr_auc', 0):.4f}", badge="Eval", accent="#6366F1", icon="📈")

        st.caption(
            f"Default classification threshold: **{eval_json.get('default_threshold', 0.50):.2f}**. "
            "Continuous Churn_Probability is used for operational risk-tiering."
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # -----------------------------------------------------------------------
        # Saved evaluation figures
        # -----------------------------------------------------------------------
        fig_col1, fig_col2, fig_col3 = st.columns(3)

        with fig_col1:
            st.markdown("**Confusion Matrix**")
            if CONFUSION_MATRIX_PATH.exists():
                st.image(str(CONFUSION_MATRIX_PATH), use_container_width=True)
            else:
                st.info("Confusion matrix image not found.")

        with fig_col2:
            st.markdown("**ROC Curve**")
            if ROC_CURVE_PATH.exists():
                st.image(str(ROC_CURVE_PATH), use_container_width=True)
            else:
                st.info("ROC curve image not found.")

        with fig_col3:
            st.markdown("**Precision-Recall Curve**")
            if PR_CURVE_PATH.exists():
                st.image(str(PR_CURVE_PATH), use_container_width=True)
            else:
                st.info("Precision-recall curve image not found.")

    st.markdown("<br>", unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Section 4 — Threshold analysis
    # -----------------------------------------------------------------------
    st.markdown("### ⚖️ Classification Threshold Trade-off Analysis")

    if eval_json and "threshold_analysis" in eval_json:
        thresh_df = pd.DataFrame(eval_json["threshold_analysis"])
        thresh_df.columns = ["Threshold", "Precision", "Recall", "F1"]

        ta_left, ta_right = st.columns([1, 2])

        with ta_left:
            st.dataframe(
                thresh_df.style.format(
                    {
                        "Threshold": "{:.2f}",
                        "Precision": "{:.4f}",
                        "Recall": "{:.4f}",
                        "F1": "{:.4f}",
                    }
                ),
                use_container_width=True,
                height=340,
            )

        with ta_right:
            fig_thresh = go.Figure()
            fig_thresh.add_trace(
                go.Scatter(
                    x=thresh_df["Threshold"],
                    y=thresh_df["Precision"],
                    name="Precision",
                    mode="lines+markers",
                    line=dict(color="#3B82F6", width=2),
                )
            )
            fig_thresh.add_trace(
                go.Scatter(
                    x=thresh_df["Threshold"],
                    y=thresh_df["Recall"],
                    name="Recall",
                    mode="lines+markers",
                    line=dict(color="#EF4444", width=2),
                )
            )
            fig_thresh.add_trace(
                go.Scatter(
                    x=thresh_df["Threshold"],
                    y=thresh_df["F1"],
                    name="F1 Score",
                    mode="lines+markers",
                    line=dict(color="#10B981", width=2),
                )
            )
            fig_thresh.add_vline(
                x=0.50,
                line_dash="dash",
                line_color="#94A3B8",
                annotation_text="Default (0.50)",
                annotation_position="top right",
            )
            fig_thresh.update_layout(
                xaxis_title="Classification Threshold",
                yaxis_title="Metric Value",
                yaxis_range=[0, 1],
                hovermode="x unified",
            )
            apply_chart_style(fig_thresh, "Precision / Recall / F1 Trade-off Curves")
            st.plotly_chart(fig_thresh, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Section 5 — Model coefficient associations
    # -----------------------------------------------------------------------
    st.markdown("### 🧬 Feature Risk Drivers (Standardised Coefficients)")

    if eval_json and "feature_coefficients" in eval_json:
        coef_df = pd.DataFrame(eval_json["feature_coefficients"])
        coef_df = coef_df.sort_values("coefficient")
        coef_df["direction"] = coef_df["coefficient"].apply(
            lambda v: "Higher Inactivity Risk (+)" if v > 0 else "Lower Inactivity Risk (-)"
        )

        fig_coef = px.bar(
            coef_df,
            x="coefficient",
            y="feature",
            orientation="h",
            color="direction",
            color_discrete_map={
                "Higher Inactivity Risk (+)": "#EF4444",
                "Lower Inactivity Risk (-)": "#10B981",
            },
            labels={"coefficient": "Standardised Coefficient", "feature": "Feature"},
            text="coefficient",
        )
        fig_coef.update_traces(texttemplate="%{x:.3f}", textposition="outside")
        fig_coef.add_vline(x=0, line_dash="solid", line_color="#64748B", line_width=1)
        fig_coef.update_layout(
            legend_title_text="",
            yaxis_title="",
        )
        apply_chart_style(fig_coef, "Standardised Feature Coefficients")
        st.plotly_chart(fig_coef, use_container_width=True)
