"""
app/pages/executive.py — Page 1: Executive Dashboard

Displays:
  - 5 Glassmorphic KPI Cards
  - Monthly revenue trend ( observation period ) with smooth area plot
  - RFM segment distribution
  - Risk-tier distribution
  - Revenue by country (top 10)
  - Key Insights panel
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.utils import fmt_gbp, load_transactions, RISK_PALETTE, SEGMENT_PALETTE, apply_chart_style, render_kpi_card


def render(df: pd.DataFrame, model) -> None:
    if df is None or len(df) == 0:
        st.warning("No customer data available.")
        return

    # -----------------------------------------------------------------------
    # KPI computation
    # -----------------------------------------------------------------------
    tx = load_transactions()
    total_identified = int(tx["CustomerID"].nunique()) if tx is not None else "N/A"
    eligible_customers = len(df)
    historical_revenue = df["Monetary"].sum()

    has_risk = "Risk_Tier" in df.columns
    n_high_risk = int((df["Risk_Tier"] == "High Risk").sum()) if has_risk else 0
    high_risk_revenue = (
        df.loc[df["Risk_Tier"] == "High Risk", "Monetary"].sum() if has_risk else 0.0
    )

    # -----------------------------------------------------------------------
    # KPI Glassmorphic Cards
    # -----------------------------------------------------------------------
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        render_kpi_card(
            "Total Identified",
            f"{total_identified:,}" if isinstance(total_identified, int) else str(total_identified),
            subtitle="Raw transaction base",
            badge="Baseline",
            accent="#3B82F6",
            icon="👥",
        )
    with col2:
        render_kpi_card(
            "Eligible Cohort",
            f"{eligible_customers:,}",
            subtitle="≥2 orders & ≥30 days span",
            badge="Filtered",
            accent="#8B5CF6",
            icon="🎯",
        )
    with col3:
        render_kpi_card(
            "Historical Revenue",
            fmt_gbp(historical_revenue),
            subtitle="Observation period total",
            badge="GBP (£)",
            accent="#10B981",
            icon="💰",
        )
    with col4:
        render_kpi_card(
            "High-Risk Base",
            f"{n_high_risk:,}",
            subtitle=f"{(n_high_risk/eligible_customers*100):.1f}% of cohort",
            badge="Top 20%",
            accent="#EF4444",
            icon="⚠️",
        )
    with col5:
        render_kpi_card(
            "At-Risk Revenue",
            fmt_gbp(high_risk_revenue),
            subtitle=f"{(high_risk_revenue/historical_revenue*100):.1f}% of total rev",
            badge="Priority Target",
            accent="#F59E0B",
            icon="🚨",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Charts — Row 1: Monthly Revenue Area Trend + RFM Segment Distribution
    # -----------------------------------------------------------------------
    row1_left, row1_right = st.columns([3, 2])

    with row1_left:
        if tx is not None:
            tx["YearMonth"] = tx["InvoiceDate"].dt.to_period("M").dt.to_timestamp()
            monthly = (
                tx.groupby("YearMonth")["Revenue"]
                .sum()
                .reset_index()
                .rename(columns={"Revenue": "Revenue (£)"})
            )
            fig_trend = px.area(
                monthly,
                x="YearMonth",
                y="Revenue (£)",
                markers=True,
                labels={"YearMonth": "Month"},
            )
            fig_trend.update_traces(
                line_color="#6366F1",
                fillcolor="rgba(99, 102, 241, 0.15)",
                marker=dict(size=7, color="#818CF8", line=dict(width=2, color="#6366F1")),
            )
            fig_trend.update_layout(
                yaxis_tickprefix="£",
                yaxis_tickformat=",.0f",
                xaxis_title="",
                yaxis_title="Revenue (£)",
                hovermode="x unified",
            )
            apply_chart_style(fig_trend, "📈 Monthly Revenue Trajectory (GBP)")
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("Transaction data not available for trend chart.")

    with row1_right:
        if "RFM_Segment" in df.columns:
            seg_counts = (
                df["RFM_Segment"]
                .value_counts()
                .reset_index()
                .rename(columns={"RFM_Segment": "Segment", "count": "Customers"})
            )
            ordered = ["Very High Value", "High Value", "Medium Value", "Lower Value"]
            seg_counts["Segment"] = pd.Categorical(seg_counts["Segment"], categories=ordered, ordered=True)
            seg_counts = seg_counts.sort_values("Segment")
            
            fig_seg = px.bar(
                seg_counts,
                x="Segment",
                y="Customers",
                color="Segment",
                color_discrete_map=SEGMENT_PALETTE,
                text="Customers",
            )
            fig_seg.update_traces(textposition="outside", texttemplate="%{text:,}")
            fig_seg.update_layout(
                showlegend=False,
                xaxis_title="",
            )
            apply_chart_style(fig_seg, "📊 Customer Count by RFM Segment")
            st.plotly_chart(fig_seg, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Charts — Row 2: Risk-Tier Donut + Revenue by Country
    # -----------------------------------------------------------------------
    row2_left, row2_right = st.columns([2, 3])

    with row2_left:
        if has_risk:
            tier_counts = (
                df["Risk_Tier"]
                .value_counts()
                .reset_index()
                .rename(columns={"Risk_Tier": "Risk Tier", "count": "Customers"})
            )
            ordered_tiers = ["High Risk", "Medium Risk", "Low Risk"]
            tier_counts["Risk Tier"] = pd.Categorical(
                tier_counts["Risk Tier"], categories=ordered_tiers, ordered=True
            )
            tier_counts = tier_counts.sort_values("Risk Tier")
            fig_tier = px.pie(
                tier_counts,
                names="Risk Tier",
                values="Customers",
                color="Risk Tier",
                color_discrete_map=RISK_PALETTE,
                hole=0.55,
            )
            fig_tier.update_traces(
                textposition="inside",
                textinfo="percent+label",
                marker=dict(line=dict(color="#0F172A", width=2)),
            )
            fig_tier.update_layout(
                showlegend=False,
            )
            apply_chart_style(fig_tier, "🎯 Customer Churn Risk Tiers")
            st.plotly_chart(fig_tier, use_container_width=True)

    with row2_right:
        if tx is not None and "Country" in tx.columns:
            country_rev = (
                tx.groupby("Country")["Revenue"]
                .sum()
                .sort_values(ascending=False)
                .head(10)
                .reset_index()
                .rename(columns={"Revenue": "Revenue (£)"})
            )
            fig_country = px.bar(
                country_rev,
                x="Revenue (£)",
                y="Country",
                orientation="h",
                color="Revenue (£)",
                color_continuous_scale="Viridis",
                labels={"Country": ""},
                text="Revenue (£)",
            )
            fig_country.update_traces(texttemplate="£%{x:,.0f}", textposition="outside")
            fig_country.update_layout(
                yaxis=dict(autorange="reversed"),
                xaxis_tickprefix="£",
                xaxis_tickformat=",.0f",
                coloraxis_showscale=False,
            )
            apply_chart_style(fig_country, "🌍 Top 10 Revenue Generating Markets")
            st.plotly_chart(fig_country, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Key Insights Panel
    # -----------------------------------------------------------------------
    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
            border: 1px solid rgba(99, 102, 241, 0.25);
            border-radius: 12px;
            padding: 20px 24px;
        ">
            <h4 style="color: #F8FAFC; margin-top: 0; margin-bottom: 12px; font-weight: 700;">💡 Executive Key Insights</h4>
        """,
        unsafe_allow_html=True,
    )

    churn_rate = df["Churn_90D"].mean() if "Churn_90D" in df.columns else None
    high_risk_pct = n_high_risk / eligible_customers * 100 if eligible_customers else 0
    high_risk_rev_pct = (
        high_risk_revenue / historical_revenue * 100 if historical_revenue else 0
    )

    insights = []
    if churn_rate is not None:
        insights.append(
            f"<b>{churn_rate * 100:.1f}% Inactivity Rate:</b> {churn_rate * 100:.1f}% of eligible customers had no qualifying purchase in the 90-day post-observation window."
        )
    insights.append(
        f"<b>High-Risk Concentration:</b> {n_high_risk:,} customers ({high_risk_pct:.1f}% of base) are categorized under <b>High Risk</b> tier."
    )
    insights.append(
        f"<b>Revenue Vulnerability:</b> High-Risk customers contribute <b>{fmt_gbp(high_risk_revenue)}</b> ({high_risk_rev_pct:.1f}% of historical sales)."
    )
    if "RFM_Segment" in df.columns:
        top_seg = df["RFM_Segment"].value_counts().idxmax()
        top_seg_n = df["RFM_Segment"].value_counts().max()
        insights.append(
            f"<b>Dominant Segment:</b> Largest RFM segment is <b>{top_seg}</b> with {top_seg_n:,} customers ({top_seg_n / eligible_customers * 100:.1f}% of total)."
        )

    for ins in insights:
        st.markdown(f"<div style='color: #CBD5E1; margin-bottom: 8px;'>• {ins}</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
