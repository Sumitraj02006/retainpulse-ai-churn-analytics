"""
tests/data/test_load.py
------------------------
Tests for src/data/load.py — schema validation and date parsing.
"""

import pandas as pd
import pytest

from src.data.load import _validate_schema
from src.config import EXPECTED_COLUMNS


def _make_df(cols=None):
    """Return a minimal DataFrame with the expected schema."""
    cols = cols or EXPECTED_COLUMNS
    data = {c: [] for c in cols}
    return pd.DataFrame(data)


def test_schema_valid():
    """No exception raised when all expected columns are present."""
    df = _make_df()
    _validate_schema(df)  # should not raise


def test_schema_missing_column():
    """ValueError raised when a required column is absent."""
    partial_cols = [c for c in EXPECTED_COLUMNS if c != "CustomerID"]
    df = _make_df(partial_cols)
    with pytest.raises(ValueError, match="CustomerID"):
        _validate_schema(df)


def test_schema_extra_column_ok():
    """Extra columns beyond expected are acceptable."""
    df = _make_df(EXPECTED_COLUMNS + ["ExtraCol"])
    _validate_schema(df)  # should not raise
