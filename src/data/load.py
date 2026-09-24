"""
src/data/load.py
-----------------
Responsible for:
  - loading data/raw/Online Retail.xlsx
  - validating the expected column schema
  - parsing InvoiceDate as datetime
  - loading data/processed/qualified_transactions.csv (Phase 3+)
"""

import pandas as pd

from src.config import (
    EXPECTED_COLUMNS,
    QUALIFIED_TRANSACTIONS_PATH,
    RAW_DATA_PATH,
    RAW_SHEET_NAME,
    SNAPSHOT_DATE,
)


def load_raw(path=None, sheet=None) -> pd.DataFrame:
    """Load the raw Online Retail workbook and return a validated DataFrame.

    Parameters
    ----------
    path : path-like, optional
        Override for the default raw data path.
    sheet : str, optional
        Override for the worksheet name.

    Returns
    -------
    pd.DataFrame
        Raw transactions with InvoiceDate parsed as datetime.

    Raises
    ------
    FileNotFoundError
        If the workbook does not exist at the specified path.
    ValueError
        If any expected column is absent from the workbook.
    """
    path = path or RAW_DATA_PATH
    sheet = sheet or RAW_SHEET_NAME

    if not pd.io.common.file_exists(str(path)):
        raise FileNotFoundError(
            f"Raw dataset not found at '{path}'. "
            "Ensure 'data/raw/Online Retail.xlsx' exists."
        )

    df = pd.read_excel(path, sheet_name=sheet, dtype={"CustomerID": "float64"})

    _validate_schema(df)

    # Parse InvoiceDate — not a native datetime in the Excel file
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")

    return df


def load_qualified_transactions(path=None) -> pd.DataFrame:
    """Load the processed qualified_transactions.csv produced by Phase 2.

    Parameters
    ----------
    path : path-like, optional
        Override for the default QUALIFIED_TRANSACTIONS_PATH.

    Returns
    -------
    pd.DataFrame
        Qualified transactions with InvoiceDate as datetime and
        CustomerID as float64.

    Raises
    ------
    FileNotFoundError
        If the CSV does not exist at the specified path.
    """
    path = path or QUALIFIED_TRANSACTIONS_PATH

    if not pd.io.common.file_exists(str(path)):
        raise FileNotFoundError(
            f"Qualified transactions not found at '{path}'. "
            "Run notebook 02_data_cleaning.ipynb first."
        )

    df = pd.read_csv(
        path,
        dtype={"CustomerID": "float64"},
        parse_dates=["InvoiceDate"],
    )
    return df


def _validate_schema(df: pd.DataFrame) -> None:
    """Raise ValueError if any expected column is missing."""
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Raw workbook is missing expected columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )
