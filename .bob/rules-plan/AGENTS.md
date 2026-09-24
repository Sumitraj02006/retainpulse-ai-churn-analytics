# AGENTS.md — Plan Mode (Architecture & Design)

This file provides guidance to agents planning or designing changes in this repository.

## Architectural Constraints

- **Dependency flow is strictly one-way**: raw data → loading/cleaning → feature engineering → model training/evaluation → application. Lower-level stages must not depend on later-stage artifacts.
- **Snapshot date is a hard constant**: `SNAPSHOT_DATE = 2011-08-31`. It is NOT derived from data; it is NOT configurable.
- **Churn label creation is separate from feature engineering**: `Churn_90D` uses the future window (01-Sep–29-Nov-2011); RFM features use only the observation period (up to 31-Aug-2011). These must not be mixed in the same transformation step.
- **Leakage rule**: No predictor feature may reference any transaction after 31-Aug-2011. The future window feeds only `Churn_90D`.
- **Primary model**: Logistic Regression in a scikit-learn `Pipeline`. The pipeline must encapsulate all preprocessing so the fitted pipeline can be saved with `joblib.dump()` and loaded in the Streamlit app without re-fitting.
- **Feature engineering is stateful**: RFM scaler/quantile boundaries fit on train data must be persisted inside the pipeline artifact — not recomputed at inference.

## Design Decisions

- **No Random Forest or XGBoost as the primary model** — spec mandates Logistic Regression for interpretability.
- **Evaluation suite is fixed**: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, confusion matrix, ROC curve, PR curve — all required.
- **SMOTE / oversampling**: if used, apply only within cross-validation folds, never before the train/test split.

## Module Boundaries

| Module | Responsibility |
|--------|---------------|
| `src/data/load.py` | Read `data/raw/Online Retail.xlsx`, parse dates, apply qualifying-purchase filter |
| `src/data/clean.py` | Customer eligibility filter (≥ 2 invoices, ≥ 30-day span) |
| `src/features/rfm.py` | Compute RFM features against `SNAPSHOT_DATE`; sklearn-compatible fit/transform |
| `src/features/churn_label.py` | Assign `Churn_90D` using the 01-Sep–29-Nov-2011 future window |
| `src/models/train.py` | Build and fit Logistic Regression Pipeline; persist with joblib |
| `src/models/evaluate.py` | Full evaluation suite (all required metrics + curves) |
| `app/app.py` | Streamlit entry point; loads persisted pipeline artifact for inference |
