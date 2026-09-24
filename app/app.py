"""
app/app.py
-----------
Entry point: streamlit run app/app.py

RetainPulse AI — E-Commerce Customer Churn Analytics & Retention Decision Support System
IBM SkillsBuild Data Analytics & AI Internship 2026 — Sumit Raj
"""

import sys
import pathlib

# Ensure repo root is on the Python path when running via `streamlit run`
_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st
from app.utils import load_enriched_customers, load_model

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="RetainPulse AI — Churn & Retention Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS Injection for Modern Premium Aesthetics
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif;
}

/* App Header styling */
.main .block-container {
    padding-top: 1.8rem;
    padding-bottom: 2rem;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F172A 0%, #0B0F19 100%);
    border-right: 1px solid rgba(255, 255, 255, 0.08);
}

.sidebar-brand {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(139, 92, 246, 0.15) 100%);
    border: 1px solid rgba(99, 102, 241, 0.3);
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 16px;
    text-align: center;
}

.sidebar-title {
    background: linear-gradient(135deg, #818CF8 0%, #C084FC 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 1.4rem;
    font-weight: 800;
    margin: 0;
    letter-spacing: -0.5px;
}

.sidebar-subtitle {
    color: #94A3B8;
    font-size: 0.75rem;
    font-weight: 600;
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.author-badge {
    display: inline-block;
    background: rgba(16, 185, 129, 0.15);
    color: #34D399;
    border: 1px solid rgba(16, 185, 129, 0.3);
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    margin-top: 8px;
}

/* Glassmorphism Containers */
div[data-testid="stMetricValue"] {
    font-weight: 800 !important;
    color: #F8FAFC !important;
}

/* Styled radio buttons */
div[role="radiogroup"] label {
    background: rgba(21, 29, 48, 0.5);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 8px;
    padding: 8px 14px;
    margin-bottom: 6px;
    transition: all 0.2s ease;
}

div[role="radiogroup"] label:hover {
    border-color: rgba(99, 102, 241, 0.5);
    background: rgba(99, 102, 241, 0.1);
}

/* Top Navigation Banner */
.header-banner {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.9) 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-left: 4px solid #6366F1;
    border-radius: 12px;
    padding: 18px 24px;
    margin-bottom: 24px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.header-title {
    font-size: 1.5rem;
    font-weight: 800;
    color: #F8FAFC;
    margin: 0;
}

.header-subtitle {
    color: #94A3B8;
    font-size: 0.85rem;
    margin-top: 2px;
}

.status-tag {
    background: rgba(99, 102, 241, 0.2);
    color: #A5B4FC;
    border: 1px solid rgba(99, 102, 241, 0.4);
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Data / model loading (cached via app.utils)
# ---------------------------------------------------------------------------

df = load_enriched_customers()   # features + Churn_Probability + Risk_Tier + Retention_Priority
model = load_model()             # cached sklearn Pipeline

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

PAGES = [
    "Executive Dashboard",
    "Customer & RFM Analytics",
    "Churn / Risk Analytics",
    "Customer Risk Explorer",
]

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-title">⚡ RetainPulse AI</div>
            <div class="sidebar-subtitle">Churn & Retention Analytics</div>
            <div class="author-badge">By Sumit Raj · IBM SkillsBuild & IBM Bob</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown("### Navigation")
    page = st.radio("Select View", PAGES, index=0, label_visibility="collapsed")
    
    st.markdown("---")
    st.markdown(
        """
        <div style="color: #64748B; font-size: 0.75rem; text-align: center;">
            Observation Period: <b>Through 31-Aug-2011</b><br>
            Label Window: <b>90-Day Inactivity</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Render Header Banner
# ---------------------------------------------------------------------------

st.markdown(
    f"""
    <div class="header-banner">
        <div>
            <div class="header-title">RetainPulse AI <span style="font-weight: 400; color: #64748B;">|</span> {page}</div>
            <div class="header-subtitle">IBM SkillsBuild & IBM Bob AI Project · Sumit Raj</div>
        </div>
        <div class="status-tag">⚡ Live Analytics</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Page Routing
# ---------------------------------------------------------------------------

if page == "Executive Dashboard":
    from app.pages.executive import render
    render(df, model)

elif page == "Customer & RFM Analytics":
    from app.pages.rfm_analytics import render
    render(df)

elif page == "Churn / Risk Analytics":
    from app.pages.churn_analytics import render
    render(df, model)

elif page == "Customer Risk Explorer":
    from app.pages.risk_explorer import render
    render(df)
