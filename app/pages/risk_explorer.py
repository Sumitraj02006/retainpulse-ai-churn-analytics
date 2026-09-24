"""
app/pages/risk_explorer.py — Page 4: Customer Risk Explorer

Allows search by CustomerID and displays the full analytical profile:
  CustomerID → behavioural metrics → RFM → predicted probability → risk tier → retention priority.

Filters: Country, Risk Tier, RFM Segment.
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from app.utils import fmt_gbp, lookup_customer, RISK_PALETTE, apply_chart_style, render_kpi_card

_DISPLAY_COLS = [
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


def render(df: pd.DataFrame) -> None:
    if df is None or len(df) == 0:
        st.warning("No customer data available.")
        return

    # -----------------------------------------------------------------------
    # Sidebar: global filters + search
    # -----------------------------------------------------------------------
    st.sidebar.markdown("### 🔎 Explorer Controls")

    search_input = st.sidebar.text_input(
        "Search by CustomerID",
        placeholder="e.g. 12347",
    )

    all_countries = sorted(df["Country"].dropna().unique().tolist())
    filter_countries = st.sidebar.multiselect(
        "Country Filter", options=all_countries, default=[], placeholder="All countries"
    )

    risk_options = ["High Risk", "Medium Risk", "Low Risk"]
    filter_risks = st.sidebar.multiselect(
        "Risk Tier Filter", options=risk_options, default=[], placeholder="All tiers"
    )

    seg_options = ["Very High Value", "High Value", "Medium Value", "Lower Value"]
    filter_segs = st.sidebar.multiselect(
        "RFM Segment Filter", options=seg_options, default=[], placeholder="All segments"
    )

    # -----------------------------------------------------------------------
    # Customer ID search
    # -----------------------------------------------------------------------
    if search_input.strip():
        result = lookup_customer(df, search_input.strip())
        if len(result) == 0:
            st.warning(
                f"No customer found with CustomerID **{search_input.strip()}**. Please verify the ID and try again."
            )
        else:
            _render_customer_profile(result)
        return

    # -----------------------------------------------------------------------
    # Apply table filters
    # -----------------------------------------------------------------------
    mask = pd.Series(True, index=df.index)
    if filter_countries:
        mask &= df["Country"].isin(filter_countries)
    if filter_risks and "Risk_Tier" in df.columns:
        mask &= df["Risk_Tier"].isin(filter_risks)
    if filter_segs and "RFM_Segment" in df.columns:
        mask &= df["RFM_Segment"].isin(filter_segs)

    filtered = df[mask]

    if len(filtered) == 0:
        st.warning("No customers match the selected filters. Try broadening the filter criteria.")
        return

    # -----------------------------------------------------------------------
    # Filter Metrics
    # -----------------------------------------------------------------------
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_kpi_card("Matching Customers", f"{len(filtered):,}", subtitle=f"Out of {len(df):,} base", badge="Filter", accent="#3B82F6", icon="👥")
    with m2:
        n_high = int((filtered["Risk_Tier"] == "High Risk").sum()) if "Risk_Tier" in filtered.columns else 0
        render_kpi_card("High-Risk Count", f"{n_high:,}", subtitle="Top 20% risk tier", badge="High Risk", accent="#EF4444", icon="⚠️")
    with m3:
        n_med = int((filtered["Risk_Tier"] == "Medium Risk").sum()) if "Risk_Tier" in filtered.columns else 0
        render_kpi_card("Medium-Risk Count", f"{n_med:,}", subtitle="Middle risk tier", badge="Medium Risk", accent="#F59E0B", icon="⚡")
    with m4:
        n_low = int((filtered["Risk_Tier"] == "Low Risk").sum()) if "Risk_Tier" in filtered.columns else 0
        render_kpi_card("Low-Risk Count", f"{n_low:,}", subtitle="Safe tier", badge="Low Risk", accent="#10B981", icon="🛡️")

    st.markdown("<br>", unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Full data table
    # -----------------------------------------------------------------------
    avail_cols = [c for c in _DISPLAY_COLS if c in filtered.columns]
    display_df = filtered[avail_cols].copy()

    if "Monetary" in display_df.columns:
        display_df["Monetary"] = display_df["Monetary"].map(lambda v: f"£{v:,.0f}")
    if "AvgOrderValue" in display_df.columns:
        display_df["AvgOrderValue"] = display_df["AvgOrderValue"].map(lambda v: f"£{v:,.2f}")
    if "Churn_Probability" in display_df.columns:
        display_df["Churn_Probability"] = display_df["Churn_Probability"].map(
            lambda v: f"{v:.1f}%"
        )

    st.markdown(f"### 📋 Customer Risk Directory ({len(filtered):,} rows)")
    st.dataframe(display_df.reset_index(), use_container_width=True, height=400)

    st.markdown("<br>", unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Visualisations
    # -----------------------------------------------------------------------
    v1, v2 = st.columns(2)

    with v1:
        if "Risk_Tier" in filtered.columns:
            tier_counts = (
                filtered["Risk_Tier"]
                .value_counts()
                .reindex(["High Risk", "Medium Risk", "Low Risk"])
                .dropna()
                .reset_index()
                .rename(columns={"Risk_Tier": "Risk Tier", "count": "Customers"})
            )
            fig_tier = px.bar(
                tier_counts,
                x="Risk Tier",
                y="Customers",
                color="Risk Tier",
                color_discrete_map=RISK_PALETTE,
                text="Customers",
            )
            fig_tier.update_traces(textposition="outside", texttemplate="%{text:,}")
            fig_tier.update_layout(showlegend=False, xaxis_title="")
            apply_chart_style(fig_tier, "Risk Tier Distribution (Filtered)")
            st.plotly_chart(fig_tier, use_container_width=True)

    with v2:
        if "Churn_Probability" in filtered.columns:
            plot_df = filtered.copy()
            fig_prob = px.histogram(
                plot_df,
                x="Churn_Probability",
                nbins=35,
                color="Risk_Tier" if "Risk_Tier" in plot_df.columns else None,
                color_discrete_map=RISK_PALETTE,
                category_orders={"Risk_Tier": ["Low Risk", "Medium Risk", "High Risk"]},
                labels={"Churn_Probability": "Predicted Inactivity Probability (%)"},
            )
            apply_chart_style(fig_prob, "Churn Probability Spectrum (Filtered)")
            st.plotly_chart(fig_prob, use_container_width=True)


def _render_customer_profile(row: pd.DataFrame) -> None:
    """Render a detailed profile card for a single customer."""
    cid = row.index[0]
    r = row.iloc[0]

    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(15, 23, 42, 0.95) 100%);
            border: 1px solid rgba(99, 102, 241, 0.3);
            border-radius: 12px;
            padding: 20px 24px;
            margin-bottom: 24px;
        ">
            <h3 style="color: #F8FAFC; margin-top: 0; font-weight: 800;">👤 Customer Profile — ID #{cid:.0f}</h3>
            <span style="color: #94A3B8; font-size: 0.85rem;">Country: {r.get('Country', '—')}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### 🛍️ Behavioural Overview")
    b1, b2, b3, b4, b5 = st.columns(5)
    with b1:
        render_kpi_card("Recency", f"{int(r.get('Recency', 0))}d", subtitle="Days Inactive", accent="#3B82F6", icon="⏱️")
    with b2:
        render_kpi_card("Frequency", f"{int(r.get('Frequency', 0))}", subtitle="Total Orders", accent="#8B5CF6", icon="🛒")
    with b3:
        render_kpi_card("Monetary", fmt_gbp(r.get("Monetary", 0)), subtitle="Total Revenue", accent="#10B981", icon="💰")
    with b4:
        render_kpi_card("Avg Basket", f"£{r.get('AvgOrderValue', 0):,.2f}", subtitle="Per Order", accent="#06B6D4", icon="💳")
    with b5:
        render_kpi_card("Active Months", f"{int(r.get('ActiveMonths', 0))}", subtitle="Distinct Months", accent="#F59E0B", icon="📅")

    st.markdown("#### 🏷️ RFM Segmentation & Risk Assessment")
    r1, r2, r3, r4 = st.columns(4)
    churn_prob = r.get("Churn_Probability", None)
    risk_tier = r.get("Risk_Tier", "—")
    retention = r.get("Retention_Priority", "—")
    accent_risk = RISK_PALETTE.get(risk_tier, "#6366F1")

    with r1:
        render_kpi_card("RFM Score", f"{r.get('RFM_Score', '—')}", subtitle=f"Quintiles: {r.get('R','—')}/{r.get('F','—')}/{r.get('M','—')}", accent="#8B5CF6", icon="⭐")
    with r2:
        render_kpi_card("RFM Segment", str(r.get("RFM_Segment", "—")), subtitle="Value Group", accent="#3B82F6", icon="🏷️")
    with r3:
        render_kpi_card("Churn Probability", f"{churn_prob:.1f}%" if churn_prob is not None else "—", subtitle="90D Inactivity Risk", accent=accent_risk, icon="🎯")
    with r4:
        render_kpi_card("Risk Tier", risk_tier, subtitle=retention, accent=accent_risk, icon="🚨")
