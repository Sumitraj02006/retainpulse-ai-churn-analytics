"""
Project-wide constants.  Import from here — never hard-code dates or paths inline.
"""
import pathlib

import pandas as pd

# ---------------------------------------------------------------------------
# Paths (relative to repo root)
# ---------------------------------------------------------------------------
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW_DATA_PATH = REPO_ROOT / "data" / "raw" / "Online Retail.xlsx"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
MODELS_DIR = REPO_ROOT / "models"
REPORTS_DIR = REPO_ROOT / "reports"

RAW_SHEET_NAME = "Online Retail"
MODEL_ARTIFACT_PATH = MODELS_DIR / "churn_model.joblib"
CUSTOMER_FEATURES_PATH = PROCESSED_DIR / "customer_features.csv"
QUALIFIED_TRANSACTIONS_PATH = PROCESSED_DIR / "qualified_transactions.csv"

# ---------------------------------------------------------------------------
# Temporal constants — never derive from data
# ---------------------------------------------------------------------------
SNAPSHOT_DATE = pd.Timestamp("2011-08-31")
OBSERVATION_START = pd.Timestamp("2010-12-01")
OBSERVATION_END = pd.Timestamp("2011-08-31")  # same as SNAPSHOT_DATE
FUTURE_START = pd.Timestamp("2011-09-01")
FUTURE_END = pd.Timestamp("2011-11-29")

# ---------------------------------------------------------------------------
# Expected raw schema
# ---------------------------------------------------------------------------
EXPECTED_COLUMNS = [
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country",
]
