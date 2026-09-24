"""
src/data/clean.py
------------------
Responsible for:
  - reporting data-quality issues (audit)
  - removing exact duplicate rows
  - identifying and separating cancellations
  - applying the four-condition qualifying-purchase filter
  - computing Revenue
  - customer eligibility filtering
  - saving processed outputs
"""

import json
import pathlib

import pandas as pd

from src.config import PROCESSED_DIR, SNAPSHOT_DATE


# ---------------------------------------------------------------------------
# Audit / profiling
# ---------------------------------------------------------------------------


def audit_raw(df: pd.DataFrame) -> dict:
    """Profile the raw DataFrame and return counts of each data-quality issue.

    Does NOT modify the DataFrame.  All counts are reported before any filtering.

    Returns
    -------
    dict with keys:
      raw_rows, raw_columns, missing_customer_id, missing_description,
      exact_duplicates, cancellation_rows,
      negative_quantity_rows, zero_quantity_rows,
      non_positive_quantity_rows,
      negative_unit_price_rows, zero_unit_price_rows,
      non_positive_unit_price_rows,
      date_parse_failures, date_min, date_max,
      unique_invoices, unique_products, unique_customers, country_count
    """
    report = {
        "raw_rows": len(df),
        "raw_columns": len(df.columns),
        "missing_customer_id": int(df["CustomerID"].isna().sum()),
        "missing_description": int(df["Description"].isna().sum()),
        "exact_duplicates": int(df.duplicated().sum()),
        "cancellation_rows": int(
            df["InvoiceNo"].astype(str).str.startswith("C").sum()
        ),
        "negative_quantity_rows": int((df["Quantity"] < 0).sum()),
        "zero_quantity_rows": int((df["Quantity"] == 0).sum()),
        "non_positive_quantity_rows": int((df["Quantity"] <= 0).sum()),
        "negative_unit_price_rows": int((df["UnitPrice"] < 0).sum()),
        "zero_unit_price_rows": int((df["UnitPrice"] == 0).sum()),
        "non_positive_unit_price_rows": int((df["UnitPrice"] <= 0).sum()),
        "date_parse_failures": int(df["InvoiceDate"].isna().sum()),
        "date_min": str(df["InvoiceDate"].min().date()) if not df["InvoiceDate"].isna().all() else None,
        "date_max": str(df["InvoiceDate"].max().date()) if not df["InvoiceDate"].isna().all() else None,
        "unique_invoices": int(df["InvoiceNo"].nunique()),
        "unique_products": int(df["StockCode"].nunique()),
        "unique_customers": int(df["CustomerID"].nunique()),
        "country_count": int(df["Country"].nunique()),
    }
    return report


# ---------------------------------------------------------------------------
# Backward-compatibility alias — existing code that called audit_raw and
# accessed keys like 'total_rows', 'cancellation_invoices', or
# 'non_positive_quantity' / 'non_positive_unit_price' still works.
# ---------------------------------------------------------------------------

_COMPAT_ALIASES = {
    "total_rows": "raw_rows",
    "cancellation_invoices": "cancellation_rows",
    "non_positive_quantity": "non_positive_quantity_rows",
    "non_positive_unit_price": "non_positive_unit_price_rows",
}


class _AuditReport(dict):
    """dict subclass that returns values via legacy key aliases."""

    def __getitem__(self, key):
        return super().__getitem__(_COMPAT_ALIASES.get(key, key))

    def __contains__(self, key):
        return super().__contains__(_COMPAT_ALIASES.get(key, key))

    def get(self, key, default=None):
        return super().get(_COMPAT_ALIASES.get(key, key), default)


def audit_raw(df: pd.DataFrame) -> "_AuditReport":  # noqa: F811  (intentional redefinition)
    """Profile the raw DataFrame and return counts of each data-quality issue.

    Does NOT modify the DataFrame.  All counts are reported before any filtering.

    Returns
    -------
    dict-like with keys:
      raw_rows, raw_columns, missing_customer_id, missing_description,
      exact_duplicates, cancellation_rows,
      negative_quantity_rows, zero_quantity_rows, non_positive_quantity_rows,
      negative_unit_price_rows, zero_unit_price_rows, non_positive_unit_price_rows,
      date_parse_failures, date_min, date_max,
      unique_invoices, unique_products, unique_customers, country_count

    Legacy aliases also accepted:
      total_rows → raw_rows
      cancellation_invoices → cancellation_rows
      non_positive_quantity → non_positive_quantity_rows
      non_positive_unit_price → non_positive_unit_price_rows
    """
    data = {
        "raw_rows": len(df),
        "raw_columns": len(df.columns),
        "missing_customer_id": int(df["CustomerID"].isna().sum()),
        "missing_description": int(df["Description"].isna().sum()),
        "exact_duplicates": int(df.duplicated().sum()),
        "cancellation_rows": int(
            df["InvoiceNo"].astype(str).str.startswith("C").sum()
        ),
        "negative_quantity_rows": int((df["Quantity"] < 0).sum()),
        "zero_quantity_rows": int((df["Quantity"] == 0).sum()),
        "non_positive_quantity_rows": int((df["Quantity"] <= 0).sum()),
        "negative_unit_price_rows": int((df["UnitPrice"] < 0).sum()),
        "zero_unit_price_rows": int((df["UnitPrice"] == 0).sum()),
        "non_positive_unit_price_rows": int((df["UnitPrice"] <= 0).sum()),
        "date_parse_failures": int(df["InvoiceDate"].isna().sum()),
        "date_min": str(df["InvoiceDate"].min().date()) if not df["InvoiceDate"].isna().all() else None,
        "date_max": str(df["InvoiceDate"].max().date()) if not df["InvoiceDate"].isna().all() else None,
        "unique_invoices": int(df["InvoiceNo"].nunique()),
        "unique_products": int(df["StockCode"].nunique()),
        "unique_customers": int(df["CustomerID"].nunique()),
        "country_count": int(df["Country"].nunique()),
    }
    return _AuditReport(data)


# ---------------------------------------------------------------------------
# Cleaning steps
# ---------------------------------------------------------------------------


def remove_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with exact duplicate rows removed."""
    return df.drop_duplicates().copy()


def flag_cancellations(df: pd.DataFrame) -> pd.DataFrame:
    """Add boolean column 'IsCancellation' (True where InvoiceNo starts with 'C').

    Does not remove any rows — use filter_qualifying_purchases for that.
    """
    df = df.copy()
    df["IsCancellation"] = df["InvoiceNo"].astype(str).str.startswith("C")
    return df


def filter_qualifying_purchases(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the four-condition qualifying-purchase rule and return valid rows.

    Conditions (ALL must be true):
      1. CustomerID is not null
      2. InvoiceNo does NOT start with 'C'
      3. Quantity > 0
      4. UnitPrice > 0

    Revenue is calculated after numeric validation.
    """
    mask = (
        df["CustomerID"].notna()
        & ~df["InvoiceNo"].astype(str).str.startswith("C")
        & (df["Quantity"] > 0)
        & (df["UnitPrice"] > 0)
    )
    qualified = df.loc[mask].copy()
    qualified["Revenue"] = qualified["Quantity"] * qualified["UnitPrice"]
    return qualified


# ---------------------------------------------------------------------------
# Customer eligibility
# ---------------------------------------------------------------------------


def filter_eligible_customers(
    df_qualified: pd.DataFrame, snapshot: pd.Timestamp = SNAPSHOT_DATE
) -> pd.DataFrame:
    """Return rows belonging only to eligible customers.

    Eligibility (observation period only, up to snapshot):
      - ≥ 2 distinct qualifying invoices
      - ≥ 30 days between first and last qualifying purchase

    Parameters
    ----------
    df_qualified : pd.DataFrame
        Qualifying-purchase rows (output of filter_qualifying_purchases),
        already limited to the observation period.
    snapshot : pd.Timestamp
        Snapshot date (default SNAPSHOT_DATE).

    Returns
    -------
    pd.DataFrame
        Subset of df_qualified for eligible customers only.
    """
    # Use date-level comparison so all intraday transactions on the snapshot
    # calendar day are included (snapshot = 2011-08-31 means the full day).
    snap_date = snapshot.date()
    obs = df_qualified[df_qualified["InvoiceDate"].dt.date <= snap_date].copy()

    agg = obs.groupby("CustomerID").agg(
        n_invoices=("InvoiceNo", "nunique"),
        first_purchase=("InvoiceDate", "min"),
        last_purchase=("InvoiceDate", "max"),
    )
    agg["span_days"] = (agg["last_purchase"] - agg["first_purchase"]).dt.days

    eligible_ids = agg.loc[
        (agg["n_invoices"] >= 2) & (agg["span_days"] >= 30), []
    ].index

    return df_qualified[df_qualified["CustomerID"].isin(eligible_ids)].copy()


def get_eligible_customer_ids(
    df_qualified: pd.DataFrame, snapshot: pd.Timestamp = SNAPSHOT_DATE
) -> pd.Index:
    """Return the CustomerID values of eligible customers.

    Uses date-level comparison so all intraday transactions on the snapshot
    calendar day are included.
    """
    snap_date = snapshot.date()
    obs = df_qualified[df_qualified["InvoiceDate"].dt.date <= snap_date]
    agg = obs.groupby("CustomerID").agg(
        n_invoices=("InvoiceNo", "nunique"),
        first_purchase=("InvoiceDate", "min"),
        last_purchase=("InvoiceDate", "max"),
    )
    agg["span_days"] = (agg["last_purchase"] - agg["first_purchase"]).dt.days
    return agg.loc[(agg["n_invoices"] >= 2) & (agg["span_days"] >= 30)].index


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

_QUALIFIED_OUTPUT_COLUMNS = [
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country",
    "Revenue",
    "IsCancellation",
]


def save_qualified_transactions(
    df_qualified: pd.DataFrame,
    output_dir: pathlib.Path = PROCESSED_DIR,
) -> pathlib.Path:
    """Save the qualifying purchase DataFrame to CSV.

    Only columns defined in _QUALIFIED_OUTPUT_COLUMNS (that exist in the
    DataFrame) are written.  IsCancellation is retained if present.

    Parameters
    ----------
    df_qualified : pd.DataFrame
        Output of filter_qualifying_purchases (with optional IsCancellation).
    output_dir : pathlib.Path
        Destination directory (created if absent).

    Returns
    -------
    pathlib.Path
        Path of the written CSV file.
    """
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cols_to_write = [c for c in _QUALIFIED_OUTPUT_COLUMNS if c in df_qualified.columns]
    save_path = output_dir / "qualified_transactions.csv"
    df_qualified[cols_to_write].to_csv(save_path, index=False)
    return save_path


def save_audit_json(
    raw_report: dict,
    df_qualified: pd.DataFrame,
    output_dir: pathlib.Path = PROCESSED_DIR,
) -> pathlib.Path:
    """Write a machine-readable audit summary to JSON.

    Parameters
    ----------
    raw_report : dict
        Output of audit_raw().
    df_qualified : pd.DataFrame
        Qualifying-purchase DataFrame (post-dedup, post-filter).
    output_dir : pathlib.Path
        Destination directory (created if absent).

    Returns
    -------
    pathlib.Path
        Path of the written JSON file.
    """
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        # Raw data profile
        "raw_rows": raw_report["raw_rows"],
        "raw_columns": raw_report["raw_columns"],
        "duplicate_rows": raw_report["exact_duplicates"],
        "missing_customer_id": raw_report["missing_customer_id"],
        "missing_description": raw_report["missing_description"],
        "cancellation_rows": raw_report["cancellation_rows"],
        "non_positive_quantity_rows": raw_report["non_positive_quantity_rows"],
        "non_positive_unit_price_rows": raw_report["non_positive_unit_price_rows"],
        "negative_quantity_rows": raw_report["negative_quantity_rows"],
        "zero_quantity_rows": raw_report["zero_quantity_rows"],
        "negative_unit_price_rows": raw_report["negative_unit_price_rows"],
        "zero_unit_price_rows": raw_report["zero_unit_price_rows"],
        "date_min": raw_report["date_min"],
        "date_max": raw_report["date_max"],
        "unique_invoices": raw_report["unique_invoices"],
        "unique_products": raw_report["unique_products"],
        "unique_customers": raw_report["unique_customers"],
        "country_count": raw_report["country_count"],
        # Cleaned / qualified data
        "qualified_rows": int(len(df_qualified)),
        "qualified_customers": int(df_qualified["CustomerID"].nunique()),
    }

    save_path = output_dir / "data_quality_audit.json"
    with open(save_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return save_path
