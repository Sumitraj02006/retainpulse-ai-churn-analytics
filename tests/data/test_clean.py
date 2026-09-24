"""
tests/data/test_clean.py
-------------------------
Tests for src/data/clean.py — qualifying-purchase filter, cancellation detection,
duplicate removal, revenue calculation, customer eligibility, audit_raw, and output
helpers.
"""

import json
import pathlib
import tempfile

import pandas as pd
import pytest

from src.data.clean import (
    audit_raw,
    filter_eligible_customers,
    filter_qualifying_purchases,
    flag_cancellations,
    get_eligible_customer_ids,
    remove_exact_duplicates,
    save_audit_json,
    save_qualified_transactions,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_raw(rows):
    """Build a minimal raw transactions DataFrame from a list of dicts."""
    defaults = {
        "InvoiceNo": "536365",
        "StockCode": "85123A",
        "Description": "WHITE HANGING HEART T-LIGHT HOLDER",
        "Quantity": 6,
        "InvoiceDate": pd.Timestamp("2011-01-10"),
        "UnitPrice": 2.55,
        "CustomerID": 17850.0,
        "Country": "United Kingdom",
    }
    records = [{**defaults, **r} for r in rows]
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Qualifying-purchase filter
# ---------------------------------------------------------------------------


def test_qualifying_filter_passes_valid_row():
    df = _make_raw([{}])
    result = filter_qualifying_purchases(df)
    assert len(result) == 1


def test_qualifying_filter_removes_missing_customer():
    df = _make_raw([{"CustomerID": float("nan")}])
    result = filter_qualifying_purchases(df)
    assert len(result) == 0


def test_qualifying_filter_removes_cancellation():
    df = _make_raw([{"InvoiceNo": "C536365"}])
    result = filter_qualifying_purchases(df)
    assert len(result) == 0


def test_qualifying_filter_removes_non_positive_quantity():
    df = _make_raw([{"Quantity": 0}, {"Quantity": -1}])
    result = filter_qualifying_purchases(df)
    assert len(result) == 0


def test_qualifying_filter_removes_non_positive_unit_price():
    df = _make_raw([{"UnitPrice": 0.0}, {"UnitPrice": -0.5}])
    result = filter_qualifying_purchases(df)
    assert len(result) == 0


def test_qualifying_filter_all_conditions_must_pass():
    """A row that fails ANY one condition is excluded."""
    rows = [
        {},                                      # valid
        {"CustomerID": float("nan")},            # no customer
        {"InvoiceNo": "C1"},                     # cancellation
        {"Quantity": -1},                        # bad quantity
        {"UnitPrice": 0},                        # bad price
    ]
    df = _make_raw(rows)
    result = filter_qualifying_purchases(df)
    assert len(result) == 1


# ---------------------------------------------------------------------------
# Revenue calculation
# ---------------------------------------------------------------------------


def test_revenue_calculation():
    df = _make_raw([{"Quantity": 3, "UnitPrice": 10.0}])
    result = filter_qualifying_purchases(df)
    assert pytest.approx(result.iloc[0]["Revenue"]) == 30.0


# ---------------------------------------------------------------------------
# Cancellation detection
# ---------------------------------------------------------------------------


def test_flag_cancellations():
    df = _make_raw([{"InvoiceNo": "C999"}, {"InvoiceNo": "536365"}])
    result = flag_cancellations(df)
    assert bool(result.iloc[0]["IsCancellation"]) is True
    assert bool(result.iloc[1]["IsCancellation"]) is False


# ---------------------------------------------------------------------------
# Duplicate removal
# ---------------------------------------------------------------------------


def test_remove_exact_duplicates():
    df = _make_raw([{}, {}])  # two identical rows
    result = remove_exact_duplicates(df)
    assert len(result) == 1


def test_remove_exact_duplicates_keeps_distinct():
    df = _make_raw([{"InvoiceNo": "A"}, {"InvoiceNo": "B"}])
    result = remove_exact_duplicates(df)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# Customer eligibility
# ---------------------------------------------------------------------------


def _make_eligible_df():
    """Return a qualified DataFrame with one eligible customer (2 invoices, 40-day span)."""
    rows = [
        {
            "InvoiceNo": "INV001",
            "CustomerID": 100.0,
            "InvoiceDate": pd.Timestamp("2011-01-01"),
            "Quantity": 1,
            "UnitPrice": 1.0,
            "Revenue": 1.0,
        },
        {
            "InvoiceNo": "INV002",
            "CustomerID": 100.0,
            "InvoiceDate": pd.Timestamp("2011-02-10"),
            "Quantity": 2,
            "UnitPrice": 2.0,
            "Revenue": 4.0,
        },
    ]
    return pd.DataFrame(rows)


def test_eligible_customer_passes():
    df = _make_eligible_df()
    ids = get_eligible_customer_ids(df, snapshot=pd.Timestamp("2011-08-31"))
    assert 100.0 in ids


def test_ineligible_customer_one_invoice():
    """Customer with only 1 invoice is excluded."""
    rows = [
        {
            "InvoiceNo": "INV001",
            "CustomerID": 200.0,
            "InvoiceDate": pd.Timestamp("2011-01-01"),
            "Quantity": 1,
            "UnitPrice": 1.0,
            "Revenue": 1.0,
        }
    ]
    df = pd.DataFrame(rows)
    ids = get_eligible_customer_ids(df, snapshot=pd.Timestamp("2011-08-31"))
    assert 200.0 not in ids


def test_ineligible_customer_span_too_short():
    """Customer with < 30-day span is excluded."""
    rows = [
        {
            "InvoiceNo": "INV001",
            "CustomerID": 300.0,
            "InvoiceDate": pd.Timestamp("2011-01-01"),
            "Quantity": 1,
            "UnitPrice": 1.0,
            "Revenue": 1.0,
        },
        {
            "InvoiceNo": "INV002",
            "CustomerID": 300.0,
            "InvoiceDate": pd.Timestamp("2011-01-15"),  # only 14 days later
            "Quantity": 1,
            "UnitPrice": 1.0,
            "Revenue": 1.0,
        },
    ]
    df = pd.DataFrame(rows)
    ids = get_eligible_customer_ids(df, snapshot=pd.Timestamp("2011-08-31"))
    assert 300.0 not in ids


# ---------------------------------------------------------------------------
# audit_raw
# ---------------------------------------------------------------------------


def test_audit_raw_basic_counts():
    """audit_raw returns the correct raw row and column count."""
    df = _make_raw([{}, {}])
    report = audit_raw(df)
    assert report["raw_rows"] == 2
    assert report["raw_columns"] == len(df.columns)


def test_audit_raw_missing_customer_id():
    df = _make_raw([{"CustomerID": float("nan")}, {}])
    report = audit_raw(df)
    assert report["missing_customer_id"] == 1


def test_audit_raw_missing_description():
    df = _make_raw([{"Description": None}, {}])
    report = audit_raw(df)
    assert report["missing_description"] == 1


def test_audit_raw_exact_duplicates():
    df = _make_raw([{}, {}])  # two identical rows
    report = audit_raw(df)
    assert report["exact_duplicates"] == 1


def test_audit_raw_cancellation_rows():
    df = _make_raw([{"InvoiceNo": "C123"}, {}])
    report = audit_raw(df)
    assert report["cancellation_rows"] == 1


def test_audit_raw_non_positive_quantity():
    df = _make_raw([{"Quantity": -1}, {"Quantity": 0}, {"Quantity": 1}])
    report = audit_raw(df)
    assert report["non_positive_quantity_rows"] == 2
    assert report["negative_quantity_rows"] == 1
    assert report["zero_quantity_rows"] == 1


def test_audit_raw_non_positive_unit_price():
    df = _make_raw([{"UnitPrice": -0.5}, {"UnitPrice": 0.0}, {"UnitPrice": 2.55}])
    report = audit_raw(df)
    assert report["non_positive_unit_price_rows"] == 2
    assert report["negative_unit_price_rows"] == 1
    assert report["zero_unit_price_rows"] == 1


def test_audit_raw_date_range():
    df = _make_raw([
        {"InvoiceDate": pd.Timestamp("2011-01-01")},
        {"InvoiceDate": pd.Timestamp("2011-06-15")},
    ])
    report = audit_raw(df)
    assert report["date_min"] == "2011-01-01"
    assert report["date_max"] == "2011-06-15"


def test_audit_raw_unique_counts():
    df = _make_raw([
        {"InvoiceNo": "A", "StockCode": "X", "CustomerID": 1.0, "Country": "UK"},
        {"InvoiceNo": "B", "StockCode": "Y", "CustomerID": 2.0, "Country": "FR"},
    ])
    report = audit_raw(df)
    assert report["unique_invoices"] == 2
    assert report["unique_products"] == 2
    assert report["unique_customers"] == 2
    assert report["country_count"] == 2


def test_audit_raw_legacy_aliases():
    """Legacy key aliases still resolve without KeyError."""
    df = _make_raw([{}])
    report = audit_raw(df)
    _ = report["total_rows"]
    _ = report["cancellation_invoices"]
    _ = report["non_positive_quantity"]
    _ = report["non_positive_unit_price"]


# ---------------------------------------------------------------------------
# save_qualified_transactions / save_audit_json
# ---------------------------------------------------------------------------


def test_save_qualified_transactions_creates_csv():
    df = _make_raw([{"Quantity": 2, "UnitPrice": 5.0}])
    qualified = filter_qualifying_purchases(df)
    with tempfile.TemporaryDirectory() as tmpdir:
        out = save_qualified_transactions(qualified, output_dir=pathlib.Path(tmpdir))
        assert out.exists()
        loaded = pd.read_csv(out)
        assert len(loaded) == 1
        assert "Revenue" in loaded.columns
        assert pytest.approx(loaded.iloc[0]["Revenue"]) == 10.0


def test_save_qualified_transactions_no_missing_customer_id():
    rows = [{"CustomerID": float("nan")}, {"CustomerID": 12345.0}]
    df = _make_raw(rows)
    qualified = filter_qualifying_purchases(df)
    with tempfile.TemporaryDirectory() as tmpdir:
        out = save_qualified_transactions(qualified, output_dir=pathlib.Path(tmpdir))
        loaded = pd.read_csv(out)
        assert loaded["CustomerID"].isna().sum() == 0


def test_save_audit_json_creates_file():
    df = _make_raw([{}, {"InvoiceNo": "C1"}])
    qualified = filter_qualifying_purchases(df)
    raw_report = audit_raw(df)
    with tempfile.TemporaryDirectory() as tmpdir:
        out = save_audit_json(raw_report, qualified, output_dir=pathlib.Path(tmpdir))
        assert out.exists()
        with open(out) as f:
            audit = json.load(f)
        assert audit["raw_rows"] == 2
        assert audit["cancellation_rows"] == 1
        assert "qualified_rows" in audit
        assert "qualified_customers" in audit
