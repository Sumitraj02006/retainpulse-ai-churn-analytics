# AGENTS.md — Agent Mode (Coding)

This file provides guidance to agents when writing or modifying code in this repository.

## Critical Coding Rules

- **Never modify `data/raw/`** — write all outputs to `data/processed/` or `models/`.
- **`random_state=42`** everywhere (train/test splits, model init) for reproducibility.
- **Stratified splits**: `train_test_split(..., stratify=y)` — churn data is class-imbalanced.
- **No direct file I/O in notebooks** — data loading belongs in `src/data/load.py`; notebooks import from there.
- **Avoid silent in-place mutation**: use `.copy()` when deriving DataFrames from raw data.
- **Qualifying-purchase filter** (apply at load time, not scattered through notebooks):
  - `CustomerID` is not null
  - `InvoiceNo` does not start with `"C"`
  - `Quantity > 0`
  - `UnitPrice > 0`
- **Snapshot date is a constant**: `SNAPSHOT_DATE = pd.Timestamp("2011-08-31")` — never derive it from the data.
- **Primary model is Logistic Regression** inside a leakage-safe scikit-learn `Pipeline` — do not substitute Random Forest or XGBoost as the primary model.

## Testing

- Tests live in `tests/` mirroring `src/` structure (e.g., `tests/features/test_rfm.py` tests `src/features/rfm.py`).
- Run a single test: `pytest tests/features/test_rfm.py::test_recency_calculation -v`
- Use `pytest.approx()` for floating-point assertions on RFM scores.
- Fixture data must use a minimal synthetic DataFrame with dates bounded to the observation period.

## Import Order (isort-compatible)

```python
# 1. stdlib
import os
from datetime import datetime

# 2. third-party
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# 3. local
from src.features.rfm import compute_rfm
```
