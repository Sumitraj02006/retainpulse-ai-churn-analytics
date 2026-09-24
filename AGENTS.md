# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Project Overview

RetainPulse AI — E-Commerce Customer Churn & Retention Analytics System.
IBM SkillsBuild & IBM Bob Data Analytics & AI Internship 2026 — Sumit Raj.
Dataset: `data/raw/Online Retail.xlsx` (UCI Online Retail, UK retailer 2010–2011).
Stack: **Python 3.x, pandas, NumPy, scikit-learn, Matplotlib, Plotly, Streamlit, Joblib, OpenPyXL, Jupyter**.

## Environment Setup

```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows PowerShell

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook

# Run Streamlit app
streamlit run app/app.py
```

## Commands

| Task | Command |
|------|---------|
| Run a single notebook | `jupyter nbconvert --to notebook --execute notebooks/<name>.ipynb` |
| Lint Python files | `flake8 src/ app/ --max-line-length=100` |
| Format code | `black src/ app/ notebooks/` |
| Run tests | `pytest tests/ -v` |
| Run single test | `pytest tests/test_<module>.py::test_<name> -v` |

## Repository Structure

```
ecommerce-churn-analytics/
├── data/
│   ├── raw/              # Online Retail.xlsx — never modify
│   └── processed/        # Cleaned data and feature tables
├── notebooks/            # EDA, feature engineering, modelling notebooks
├── src/                  # Importable Python modules
├── app/                  # Streamlit dashboard (app.py entry point)
├── models/               # Persisted model artifacts (joblib)
├── assets/               # Static images, logos
├── reports/              # Figures and report outputs
├── tests/                # pytest unit tests mirroring src/ structure
├── presentation/         # Slide deck assets
├── requirements.txt
├── .gitignore
└── README.md
```

## Code Style

- **Line length**: 100 characters.
- **Formatter**: `black` — run black instead of manual formatting.
- **Imports**: stdlib → third-party → local, separated by blank lines (isort-compatible).
- **Notebooks**: Restart kernel and run all cells top-to-bottom before committing; no hidden state.
- **DataFrames**: Use `.copy()` when modifying data derived from raw; avoid silent in-place mutation.

## Churn Definition (Fixed — Not Configurable)

- **Observation period**: all transactions up to and including **31-Aug-2011** (snapshot date).
- **Future window**: **01-Sep-2011 through 29-Nov-2011** (90 days).
- **Churn_90D = 1**: eligible customer has **no qualifying purchase** in the future window.
- **Churn_90D = 0**: eligible customer has **at least one qualifying purchase** in the future window.
- **Eligible customer**: ≥ 2 distinct qualifying invoices in the observation period AND ≥ 30 days between first and last qualifying purchase.
- **Qualifying purchase**: `CustomerID` present AND `InvoiceNo` does not start with `"C"` AND `Quantity > 0` AND `UnitPrice > 0`.

## Leakage Rule (Critical)

No predictor feature may use any information after **31-Aug-2011**.
The future window (01-Sep–29-Nov-2011) is used **only** to assign `Churn_90D`; it must never feed into any feature.

## Evaluation Metrics (All Required)

Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, confusion matrix, ROC curve, precision-recall curve.

## Key Gotchas

- `CustomerID` is a float column — it will have NaN rows; drop them before any customer-level aggregation.
- `InvoiceNo` prefixed with `"C"` = cancellation; excluded by the qualifying-purchase rule.
- `Quantity ≤ 0` or `UnitPrice ≤ 0` rows are excluded by the qualifying-purchase rule — not just NaN-dropped.
- `InvoiceDate` needs `pd.to_datetime()` parsing; it is not a native datetime in the raw Excel file.
- The dataset file is `Online Retail.xlsx` (space in name) — quote the path in shell commands.
