"""
app/pages/rfm_analytics.py — Page 2: Customer & RFM Analytics

Displays:
  - Filters: Country, RFM Segment, Recency range, Monetary range
  - Recency / Frequency / Monetary / RFM-score distributions
  - RFM segment distribution
  - Recency vs Monetary scatter
  - Frequency vs Monetary scatter
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from app.utils import fmt_gbp, SEGMENT_PALETTE, apply_chart_style, render_kpi_card

_SEG_ORDER = ["Very High Value", "High Value", "Medium Value", "Lower Value"]


def render(df: pd.DataFrame) -> None:
    if df is None or len(df) == 0:
        st.warning("No customer data available.")
        return

    # -----------------------------------------------------------------------
    # Sidebar filters
    # -----------------------------------------------------------------------
    st.sidebar.markdown("### 🎛️ RFM Filter Controls")

    # Country filter
    all_countries = sorted(df["Country"].dropna().unique().tolist())
    selected_countries = st.sidebar.multiselect(
        "Country Filter",
        options=all_countries,
        default=[],
        placeholder="All countries",
    )

    # RFM Segment filter
    seg_options = [s for s in _SEG_ORDER if s in df["RFM_Segment"].unique()]
    selected_segs = st.sidebar.multiselect(
        "RFM Segment Filter",
        options=seg_options,
        default=[],
        placeholder="All segments",
    )

    # Recency range
    rec_min = int(df["Recency"].min())
    rec_max = int(df["Recency"].max())
    recency_range = st.sidebar.slider(
        "Recency (Days Inactive)",
        min_value=rec_min,
        max_value=rec_max,
        value=(rec_min, rec_max),
    )

    # Monetary range
    mon_min = float(df["Monetary"].min())
    mon_max = float(df["Monetary"].max())
    monetary_range = st.sidebar.slider(
        "Monetary Value (£)",
        min_value=mon_min,
        max_value=mon_max,
        value=(mon_min, mon_max),
        format="£%.0f",
    )

    # -----------------------------------------------------------------------
    # Apply filters
    # -----------------------------------------------------------------------
    mask = (
        (df["Recency"] >= recency_range[0])
        & (df["Recency"] <= recency_range[1])
        & (df["Monetary"] >= monetary_range[0])
        & (df["Monetary"] <= monetary_range[1])
    )
    if selected_countries:
        mask &= df["Country"].isin(selected_countries)
    if selected_segs:
        mask &= df["RFM_Segment"].isin(selected_segs)

    filtered = df[mask]

    if len(filtered) == 0:
        st.warning("No customers match the selected filters. Try broadening the filter criteria.")
        return

    # -----------------------------------------------------------------------
    # Filter KPI Summary Tiles
    # -----------------------------------------------------------------------
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_kpi_card("Filtered Cohort", f"{len(filtered):,}", subtitle=f"Out of {len(df):,} total", badge="Selection", accent="#3B82F6", icon="👥")
    with k2:
        render_kpi_card("Filtered Revenue", fmt_gbp(filtered['Monetary'].sum()), subtitle="Total Monetary", badge="Revenue", accent="#10B981", icon="💰")
    with k3:
        render_kpi_card("Avg Order Value", fmt_gbp(filtered['AvgOrderValue'].mean()), subtitle="Per Order Avg", badge="Basket", accent="#8B5CF6", icon="🛒")
    with k4:
        render_kpi_card("Mean Recency", f"{filtered['Recency'].mean():.1f} days", subtitle="Inactivity Avg", badge="Recency", accent="#F59E0B", icon="⏱️")

    st.markdown("<br>", unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Row 1 — Recency, Frequency, Monetary distributions
    # -----------------------------------------------------------------------
    st.markdown("### 📊 Distribution of Recency, Frequency & Monetary")
    c1, c2, c3 = st.columns(3)

    with c1:
        fig_rec = px.histogram(
            filtered,
            x="Recency",
            nbins=35,
            labels={"Recency": "Recency (Days)"},
        )
        fig_rec.update_traces(marker_color="#3B82F6", marker_line_color="#1E40AF", marker_line_width=1)
        apply_chart_style(fig_rec, "Recency Distribution (Days)")
        st.plotly_chart(fig_rec, use_container_width=True)

    with c2:
        fig_freq = px.histogram(
            filtered,
            x="Frequency",
            nbins=35,
            labels={"Frequency": "Order Count"},
        )
        fig_freq.update_traces(marker_color="#8B5CF6", marker_line_color="#5B21B6", marker_line_width=1)
        apply_chart_style(fig_freq, "Frequency Distribution (Orders)")
        st.plotly_chart(fig_freq, use_container_width=True)

    with c3:
        fig_mon = px.histogram(
            filtered,
            x="Monetary",
            nbins=35,
            labels={"Monetary": "Monetary (£)"},
        )
        fig_mon.update_traces(marker_color="#10B981", marker_line_color="#065F46", marker_line_width=1)
        fig_mon.update_layout(xaxis_tickprefix="£", xaxis_tickformat=",.0f")
        apply_chart_style(fig_mon, "Monetary Distribution (£)")
        st.plotly_chart(fig_mon, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Row 2 — RFM Score & Segment Distribution
    # -----------------------------------------------------------------------
    st.markdown("### 🏷️ RFM Segmentation Metrics")
    c4, c5 = st.columns(2)

    with c4:
        fig_score = px.histogram(
            filtered,
            x="RFM_Score",
            nbins=int(filtered["RFM_Score"].max() - filtered["RFM_Score"].min() + 1),
            labels={"RFM_Score": "RFM Composite Score"},
        )
        fig_score.update_traces(marker_color="#06B6D4", marker_line_color="#0891B2", marker_line_width=1)
        apply_chart_style(fig_score, "RFM Score Distribution (3 to 15)")
        st.plotly_chart(fig_score, use_container_width=True)

    with c5:
        seg_counts = (
            filtered["RFM_Segment"]
            .value_counts()
            .reindex(_SEG_ORDER)
            .dropna()
            .reset_index()
            .rename(columns={"RFM_Segment": "Segment", "count": "Customers"})
        )
        fig_seg = px.bar(
            seg_counts,
            x="Segment",
            y="Customers",
            color="Segment",
            color_discrete_map=SEGMENT_PALETTE,
            text="Customers",
        )
        fig_seg.update_traces(textposition="outside", texttemplate="%{text:,}")
        fig_seg.update_layout(showlegend=False, xaxis_title="")
        apply_chart_style(fig_seg, "RFM Segment Breakdown")
        st.plotly_chart(fig_seg, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Row 3 — Scatter Plots
    # -----------------------------------------------------------------------
    st.markdown("### 🌌 Behavioral Scatter Analysis")
    c6, c7 = st.columns(2)

    with c6:
        fig_scat1 = px.scatter(
            filtered.reset_index(),
            x="Recency",
            y="Monetary",
            color="RFM_Segment" if "RFM_Segment" in filtered.columns else None,
            color_discrete_map=SEGMENT_PALETTE,
            category_orders={"RFM_Segment": _SEG_ORDER},
            hover_data=["CustomerID", "Frequency"],
            labels={
                "Recency": "Recency (Days Inactive)",
                "Monetary": "Monetary Value (£)",
                "RFM_Segment": "Segment",
            },
            opacity=0.8,
        )
        fig_scat1.update_traces(marker=dict(size=8, line=dict(width=1, color="#0F172A")))
        fig_scat1.update_layout(
            yaxis_tickprefix="£",
            yaxis_tickformat=",.0f",
        )
        apply_chart_style(fig_scat1, "Recency vs. Monetary (£)")
        st.plotly_chart(fig_scat1, use_container_width=True)

    with c7:
        fig_scat2 = px.scatter(
            filtered.reset_index(),
            x="Frequency",
            y="Monetary",
            color="RFM_Segment" if "RFM_Segment" in filtered.columns else None,
            color_discrete_map=SEGMENT_PALETTE,
            category_orders={"RFM_Segment": _SEG_ORDER},
            hover_data=["CustomerID", "Recency"],
            labels={
                "Frequency": "Frequency (Order Count)",
                "Monetary": "Monetary Value (£)",
                "RFM_Segment": "Segment",
            },
            opacity=0.8,
        )
        fig_scat2.update_traces(marker=dict(size=8, line=dict(width=1, color="#0F172A")))
        fig_scat2.update_layout(
            yaxis_tickprefix="£",
            yaxis_tickformat=",.0f",
        )
        apply_chart_style(fig_scat2, "Frequency vs. Monetary (£)")
        st.plotly_chart(fig_scat2, use_container_width=True)
