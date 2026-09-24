# AGENTS.md — Ask Mode (Documentation & Questions)

This file provides guidance to agents answering questions about this repository.

## Project Context

- IBM SkillsBuild Data Analytics & AI Internship 2026 — Tanmay Sachin Chawhan.
- Dataset: UCI Online Retail dataset (`data/raw/Online Retail.xlsx`) — UK-based online retailer, 2010–2011.
- Goal: Predict 90-day customer churn using RFM feature engineering + Logistic Regression; surface results in a Streamlit dashboard.

## Non-Obvious Context

- **`Churn_90D` is not a native dataset label** — it is a project-defined future-inactivity proxy derived from a fixed future window (01-Sep–29-Nov-2011). The observation snapshot is fixed at 31-Aug-2011.
- **Churn threshold is not configurable** — the 90-day window and 31-Aug-2011 snapshot are hard constants, not parameters in `src/config.py`.
- **Customer eligibility has two conditions** beyond just having a CustomerID: ≥ 2 distinct qualifying invoices AND ≥ 30-day span between first and last purchase in the observation period.
- **Cancellation invoices AND zero/negative Quantity/Price rows** are all excluded by the qualifying-purchase definition — not just cancellations.
- **The dataset file name has a space**: `Online Retail.xlsx` — always quote paths referencing it.
- **Virtual environment is `.venv`**, not `venv`.
- **The project includes a Streamlit app** (`app/`) with four panels: Executive Dashboard, Customer & RFM Analytics, Churn/Risk Analytics, Customer Risk Explorer.

## Reference Notebooks Order

Intended execution order:
1. `01_eda.ipynb` — exploratory data analysis
2. `02_feature_engineering.ipynb` — qualifying-purchase filtering, RFM computation, churn label creation
3. `03_modelling.ipynb` — Logistic Regression pipeline, evaluation
4. `04_reporting.ipynb` — final visualisations and summary
