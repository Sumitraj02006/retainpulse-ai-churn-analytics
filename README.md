# RetainPulse AI — E-Commerce Customer Churn & Retention Analytics System

**IBM SkillsBuild & IBM Bob AI Internship 2026**
**Sumit Raj · Deadline: 24 September 2026**

---

## Project Overview

This project builds a complete end-to-end customer-churn analytics and retention decision-support
system for an e-commerce retailer. Using historical transaction data from the UCI Online Retail
dataset, it applies RFM (Recency, Frequency, Monetary) feature engineering and a Logistic Regression
model to rank each eligible customer's predicted future inactivity risk. The resulting risk scores and
RFM segments feed a Streamlit dashboard that surfaces actionable retention priority recommendations
for business decision-makers.

---

## Problem Statement

Customer churn is a significant cost driver in e-commerce. Retaining an existing customer is
substantially less expensive than acquiring a new one. Most transaction databases, however, do not
carry a native "churned" label. This project demonstrates how to construct a project-defined
future-inactivity proxy label from raw transaction history, engineer interpretable RFM features, train
a leakage-safe classification model, and deliver risk scores through an interactive business dashboard.

---

## Objectives

1. Clean and audit 541,909 raw transaction records from the UCI Online Retail dataset.
2. Define and apply a rigorous eligibility filter and a reproducible 90-day future-inactivity label
   (`Churn_90D`).
3. Engineer RFM and additional behavioural features for 1,765 eligible customers.
4. Train and evaluate a leakage-safe Logistic Regression pipeline.
5. Assign operational risk tiers (High / Medium / Low) across the full eligible population.
6. Deliver a four-section Streamlit dashboard with customer lookup and retention priority matrix.

---

## Dataset

### Dataset Source and Attribution

**Dataset:** UCI Online Retail
**Repository:** UCI Machine Learning Repository
**Original publication:**
> Daqing Chen, Sai Liang Sain, and Kun Guo, *Data mining for the online retail industry: A case study
> of RFM model-based customer segmentation using data mining*, Journal of Database Marketing and
> Customer Strategy Management, Vol. 19, No. 3, pp. 197â€“208, 2012.
> DOI: [10.1057/dbm.2012.17](https://doi.org/10.1057/dbm.2012.17)

**UCI page:** <https://archive.ics.uci.edu/ml/datasets/online+retail>

The dataset is publicly available for research and educational use. Place the downloaded file at:

```
data/raw/Online Retail.xlsx
```

Do **not** use a machine-specific absolute path. The file must be at exactly that relative location
within the cloned repository.

### Dataset Structure

| Column | Description |
|---|---|
| `InvoiceNo` | Invoice number â€” values starting with `"C"` are cancellations |
| `StockCode` | Product code |
| `Description` | Product description |
| `Quantity` | Units per line |
| `InvoiceDate` | Date and time of invoice |
| `UnitPrice` | Price per unit in GBP (Â£) |
| `CustomerID` | Customer identifier (float; nullable) |
| `Country` | Customer country |

**Raw data summary:**

- 541,909 transaction rows
- 8 columns
- Date range: 01-Dec-2010 through 09-Dec-2011
- 4,372 unique customer IDs (before filtering)
- 38 countries

---

## Data Cleaning

### Qualifying-Purchase Filter

Every row must satisfy all four conditions to be counted as a qualifying purchase:

1. `CustomerID` is **not null**
2. `InvoiceNo` does **not** begin with `"C"` (cancellations are excluded)
3. `Quantity > 0`
4. `UnitPrice > 0`

**Data quality findings from the raw file:**

| Issue | Count |
|---|---|
| Rows with missing CustomerID | 135,080 |
| Cancellation rows (`InvoiceNo` starts with `"C"`) | 9,288 |
| Non-positive Quantity rows | 10,624 |
| Non-positive UnitPrice rows | 2,517 |
| Duplicate rows | 5,268 |
| Missing Description | 1,454 |

After applying all filters: **392,692 qualifying purchase transactions** remain.

---

## Customer Eligibility

A customer must satisfy both conditions to be included in the modelling population:

1. **At least 2 distinct qualifying invoices** during the observation period.
2. **At least 30 days between first and last qualifying purchase** during the observation period.

These rules ensure that customers with a meaningful purchase history are modelled, and that very
new or very sparse customers are not assigned spurious churn labels.

**Eligible customers: 1,765**

---

## Feature Engineering

Features are derived exclusively from transactions in the observation period
(**01-Dec-2010 through 31-Aug-2011**). No information from the future window is used in any feature.

| Feature | Description |
|---|---|
| `Recency` | Days since last qualifying purchase before the snapshot date |
| `Frequency` | Number of distinct qualifying invoices |
| `Monetary` | Total GBP spend across all qualifying purchases |
| `AvgOrderValue` | Mean GBP spend per qualifying invoice |
| `UniqueProducts` | Distinct StockCodes purchased |
| `TotalItems` | Total units purchased |
| `ActiveMonths` | Count of distinct calendar months with at least one qualifying purchase |
| `TenureDays` | Days from customer's first qualifying purchase to the snapshot date |
| `PurchaseSpanDays` | Days between first and last qualifying purchase |

---

## RFM Methodology

Each customer receives three individual quintile scores (1â€“5):

- **R score** â€” Recency: lower recency (purchased recently) â†’ higher score
- **F score** â€” Frequency: more invoices â†’ higher score
- **M score** â€” Monetary: higher spend â†’ higher score

The composite **RFM_Score** is the sum `R + F + M` (range 3â€“15).

Customers are then grouped into four named segments:

| Segment | RFM_Score range | Count |
|---|---|---|
| Very High Value | 13â€“15 | 358 |
| High Value | 10â€“12 | 419 |
| Medium Value | 7â€“9 | 486 |
| Lower Value | 3â€“6 | 502 |

---

## Churn_90D Definition

`Churn_90D` is a **project-defined future-inactivity proxy**. It is **not** a native churn label
supplied by the UCI dataset.

| Value | Meaning |
|---|---|
| `1` | The eligible customer has **no qualifying purchase** in the future window |
| `0` | The eligible customer has **at least one qualifying purchase** in the future window |

**Temporal boundaries:**

| Window | Dates |
|---|---|
| Snapshot date | 31-Aug-2011 |
| Observation period | 01-Dec-2010 â€” 31-Aug-2011 |
| Future target window | 01-Sep-2011 â€” 29-Nov-2011 (90 days) |

**Churn_90D distribution across 1,765 eligible customers:**

| Class | Count | Percentage |
|---|---|---|
| 0 â€” Future-active | 1,275 | 72.24% |
| 1 â€” Future-inactive | 490 | 27.76% |

---

## Leakage Prevention

No predictor feature uses any information after **31-Aug-2011**. The future window
(01-Sepâ€“29-Nov-2011) is used **exclusively** to assign the `Churn_90D` label. It never feeds into
any model input feature.

All temporal constants are defined once in [`src/config.py`](src/config.py) and imported by all
notebooks and modules. Dates are never derived from the data.

The model pipeline wraps `StandardScaler` and `LogisticRegression` in a scikit-learn `Pipeline`,
ensuring that scaling parameters are fitted only on training data and applied to test data without
look-ahead.

---

## Machine Learning Methodology

**Model:** Logistic Regression
**Framework:** scikit-learn `Pipeline` (StandardScaler â†’ LogisticRegression)
**Class weight:** `balanced` (addresses the 72/28 class imbalance)
**Max iterations:** 1,000
**Random state:** 42 (fixed throughout for reproducibility)

**Train / test split:**

| Split | Size | Notes |
|---|---|---|
| Train | 1,412 | 80% of 1,765 eligible customers |
| Test | 353 | 20% stratified hold-out |

The split is stratified on `Churn_90D` to preserve class proportions. Customer IDs do not overlap
between train and test sets.

**Predictor features used in the model:**
`Recency`, `Frequency`, `Monetary`, `AvgOrderValue`, `UniqueProducts`, `TotalItems`,
`ActiveMonths`, `TenureDays`, `PurchaseSpanDays`

---

## Model Evaluation

All metrics below are from the **held-out test set (353 customers only)**.

> âš  These are **not** full-population metrics. The 1,765-customer operational risk scores are
> produced by applying the trained pipeline to the full population after evaluation, but only
> the 353-customer test set provides unbiased held-out estimates.

| Metric | Value |
|---|---|
| Accuracy | 0.5864 |
| Precision | 0.3723 |
| Recall | 0.7143 |
| F1 | 0.4895 |
| ROC-AUC | 0.7063 |
| PR-AUC | 0.4686 |
| Default threshold | 0.50 |

**Threshold analysis:**

| Threshold | Precision | Recall | F1 |
|---|---|---|---|
| 0.20 | 0.3297 | 0.9286 | 0.4866 |
| 0.30 | 0.3386 | 0.8776 | 0.4886 |
| 0.35 | 0.3529 | 0.8571 | 0.5000 |
| 0.40 | 0.3511 | 0.8061 | 0.4892 |
| 0.45 | 0.3623 | 0.7653 | 0.4918 |
| 0.50 | 0.3723 | 0.7143 | 0.4895 |
| 0.55 | 0.4061 | 0.6837 | 0.5095 |
| 0.60 | 0.4385 | 0.5816 | 0.5000 |

Evaluation artefacts (ROC curve, precisionâ€“recall curve, confusion matrix) are saved to `reports/`.

---

## Risk Segmentation

After evaluation on the held-out test set, the trained pipeline is applied to all 1,765 eligible
customers to produce operational risk scores. Risk tiers are assigned by percentile of the
predicted churn probability:

| Tier | Percentile rule | Count |
|---|---|---|
| High Risk | Top 20% (â‰¥ 80th percentile) | 353 |
| Medium Risk | 40thâ€“80th percentile | 706 |
| Low Risk | Bottom 40% (< 40th percentile) | 706 |

> These are population-wide operational risk rankings derived from model-predicted probabilities.
> They are decision-support outputs, not held-out predictions.

---

## Retention Decision Support

Each customer receives a `Retention_Priority` label based on the combination of their risk tier
and RFM segment:

| Risk Tier | RFM Segment | Retention Priority |
|---|---|---|
| High Risk | Very High Value | Priority retention attention |
| High Risk | High Value | Priority retention attention |
| High Risk | Medium Value | Targeted attention |
| High Risk | Lower Value | Lower-cost intervention |
| Medium Risk | Very High Value | Monitor / targeted engagement |
| Medium Risk | High Value | Monitor / targeted engagement |
| Medium Risk | Medium Value | Standard engagement |
| Medium Risk | Lower Value | Standard engagement |
| Low Risk | Very High Value | Relationship / loyalty attention |
| Low Risk | High Value | Relationship / loyalty attention |
| Low Risk | Medium Value | Standard engagement |
| Low Risk | Lower Value | Standard engagement |

These priority labels are **decision-support rules**, not experimentally validated intervention
policies. They encode intuitive business logic (high-value customers at high risk warrant the most
immediate attention) and should be reviewed by domain experts before operational deployment.

---

## Streamlit Dashboard

The interactive dashboard (`app/app.py`) has four sections:

1. **Executive Dashboard** â€” KPI summary cards (eligible customers, churn rate, risk tier
   distribution, top-risk count).
2. **Customer & RFM Analytics** â€” RFM segment distribution, feature distribution charts,
   transaction trend over the observation period, RFM score histogram.
3. **Churn / Risk Analytics** â€” Model evaluation metrics, ROC curve, precisionâ€“recall curve,
   confusion matrix, threshold analysis, feature coefficient analysis.
4. **Customer Risk Explorer** â€” Filterable table of all 1,765 customers with risk tiers, churn
   probability, and retention priorities. Includes a customer lookup by ID.

---

## Project Structure

```
ecommerce-churn-analytics/
â”œâ”€â”€ data/
â”‚   â”œâ”€â”€ raw/
â”‚   â”‚   â””â”€â”€ Online Retail.xlsx          # UCI source file â€” never modify
â”‚   â””â”€â”€ processed/
â”‚       â”œâ”€â”€ qualified_transactions.csv  # Post-filter transactions
â”‚       â”œâ”€â”€ customer_features.csv       # 1,765-customer RFM feature table
â”‚       â”œâ”€â”€ data_quality_audit.json     # Raw data quality statistics
â”‚       â”œâ”€â”€ rfm_churn_summary.json      # RFM/churn distribution summary
â”‚       â””â”€â”€ test_predictions.csv        # Held-out test set predictions
â”‚
â”œâ”€â”€ notebooks/
â”‚   â”œâ”€â”€ 01_data_audit.ipynb             # Raw data audit
â”‚   â”œâ”€â”€ 02_data_cleaning.ipynb          # Qualifying-purchase filter
â”‚   â”œâ”€â”€ 03_rfm_eda.ipynb                # RFM + Churn_90D feature engineering
â”‚   â””â”€â”€ 04_churn_model.ipynb            # Model training, evaluation, artefacts
â”‚
â”œâ”€â”€ src/
â”‚   â”œâ”€â”€ config.py                       # All project constants (dates, paths)
â”‚   â”œâ”€â”€ insights.py                     # Risk tiers, retention priority
â”‚   â”œâ”€â”€ data/
â”‚   â”‚   â”œâ”€â”€ load.py                     # Data loading helpers
â”‚   â”‚   â””â”€â”€ clean.py                    # Qualifying-purchase filter
â”‚   â”œâ”€â”€ features/
â”‚   â”‚   â”œâ”€â”€ rfm.py                      # RFM computation
â”‚   â”‚   â”œâ”€â”€ churn_label.py              # Churn_90D labelling
â”‚   â”‚   â””â”€â”€ build_features.py           # Feature assembly pipeline
â”‚   â””â”€â”€ models/
â”‚       â”œâ”€â”€ train.py                    # Pipeline build, train, split, risk tiers
â”‚       â””â”€â”€ evaluate.py                 # Metric computation and JSON serialisation
â”‚
â”œâ”€â”€ app/
â”‚   â”œâ”€â”€ app.py                          # Streamlit entry point
â”‚   â”œâ”€â”€ utils.py                        # Cached loaders, lookup helpers
â”‚   â””â”€â”€ pages/
â”‚       â”œâ”€â”€ executive.py                # Executive Dashboard page
â”‚       â”œâ”€â”€ rfm_analytics.py            # Customer & RFM Analytics page
â”‚       â”œâ”€â”€ churn_analytics.py          # Churn / Risk Analytics page
â”‚       â””â”€â”€ risk_explorer.py            # Customer Risk Explorer page
â”‚
â”œâ”€â”€ models/
â”‚   â””â”€â”€ churn_model.joblib              # Trained pipeline artefact
â”‚
â”œâ”€â”€ tests/
â”‚   â”œâ”€â”€ data/
â”‚   â”‚   â”œâ”€â”€ test_load.py
â”‚   â”‚   â””â”€â”€ test_clean.py
â”‚   â”œâ”€â”€ features/
â”‚   â”‚   â”œâ”€â”€ test_rfm.py
â”‚   â”‚   â””â”€â”€ test_churn_label.py
â”‚   â”œâ”€â”€ models/
â”‚   â”‚   â”œâ”€â”€ test_train.py
â”‚   â”‚   â””â”€â”€ test_evaluate.py
â”‚   â””â”€â”€ test_app_helpers.py
â”‚
â”œâ”€â”€ reports/
â”‚   â”œâ”€â”€ model_evaluation.json           # Full evaluation JSON
â”‚   â”œâ”€â”€ confusion_matrix.png
â”‚   â”œâ”€â”€ roc_curve.png
â”‚   â””â”€â”€ precision_recall_curve.png
â”‚
â”œâ”€â”€ assets/
â”‚   â””â”€â”€ screenshots/                    # Dashboard screenshots
â”‚
â”œâ”€â”€ presentation/                       # Slide deck assets
â”‚
â”œâ”€â”€ AGENTS.md                           # Agent coding guidance
â”œâ”€â”€ requirements.txt
â”œâ”€â”€ .gitignore
â”œâ”€â”€ .bobignore
â””â”€â”€ README.md
```

---

## Installation

### Prerequisites

- Python 3.10 or later
- The UCI Online Retail dataset placed at `data/raw/Online Retail.xlsx`

### 1. Clone the repository

```bash
git clone <repository-url>
cd ecommerce-churn-analytics
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

**Windows PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Verify the environment

```bash
python -m pip check
```

No broken requirements should be reported.

### 5. Obtain the dataset

Download the UCI Online Retail dataset from:
<https://archive.ics.uci.edu/ml/datasets/online+retail>

Place the file at:

```
data/raw/Online Retail.xlsx
```

---

## Usage

### Run the Streamlit dashboard

```bash
streamlit run app/app.py
```

The application loads pre-computed artefacts from `data/processed/` and `models/`. All four
dashboard sections are available immediately without re-running notebooks.

### Re-run the analysis pipeline (optional)

If you want to reproduce artefacts from scratch, execute notebooks in order:

```bash
jupyter notebook
```

1. `notebooks/01_data_audit.ipynb` â€” raw data audit
2. `notebooks/02_data_cleaning.ipynb` â€” qualifying-purchase filter, produces `qualified_transactions.csv`
3. `notebooks/03_rfm_eda.ipynb` â€” RFM feature engineering, produces `customer_features.csv`
4. `notebooks/04_churn_model.ipynb` â€” model training and evaluation, produces all `models/` and `reports/` artefacts

### Run the test suite

```bash
pytest tests/ -v
```

Expected: **141 tests passing**.

---

## Results

### Data Quality

- **541,909** raw transaction rows
- **5,268** duplicate rows removed
- **135,080** rows with missing CustomerID excluded
- **9,288** cancellation rows (InvoiceNo starts with `"C"`) excluded
- **10,624** non-positive Quantity rows excluded
- **2,517** non-positive UnitPrice rows excluded
- **392,692** qualifying purchase transactions retained

### Eligible Customer Population

- **1,765** customers met the eligibility criteria (â‰¥ 2 distinct qualifying invoices,
  â‰¥ 30 days between first and last purchase)

### Churn_90D Distribution

| Class | Count | % |
|---|---|---|
| 0 â€” Future-active | 1,275 | 72.24% |
| 1 â€” Future-inactive | 490 | 27.76% |

### RFM Segment Distribution

| Segment | Count |
|---|---|
| Very High Value | 358 |
| High Value | 419 |
| Medium Value | 486 |
| Lower Value | 502 |

### RFM Feature Summary (observation period)

| Feature | Min | Max | Mean | Median |
|---|---|---|---|---|
| Recency (days) | 0 | 236 | 54.88 | 40.0 |
| Frequency (invoices) | 2 | 127 | 5.45 | 4.0 |
| Monetary (Â£) | 21.31 | 176,355.94 | 2,555.49 | 1,132.62 |

### Model Performance (held-out test set, 353 customers)

| Metric | Value |
|---|---|
| Accuracy | 0.5864 |
| Precision | 0.3723 |
| Recall | 0.7143 |
| F1 Score | 0.4895 |
| ROC-AUC | 0.7063 |
| PR-AUC | 0.4686 |

### Operational Risk Tier Distribution (full 1,765-customer population)

| Risk Tier | Count |
|---|---|
| High Risk | 353 |
| Medium Risk | 706 |
| Low Risk | 706 |

### Model Coefficient Associations

The table below shows the Logistic Regression coefficient for each predictor.
A **negative** coefficient is **associated with lower predicted inactivity risk**; a **positive**
coefficient is **associated with higher predicted inactivity risk**. These are model associations,
not causal effects.

| Feature | Coefficient | Direction |
|---|---|---|
| ActiveMonths | âˆ’0.790 | Associated with lower predicted inactivity risk |
| Frequency | âˆ’0.774 | Associated with lower predicted inactivity risk |
| TotalItems | âˆ’0.769 | Associated with lower predicted inactivity risk |
| UniqueProducts | âˆ’0.410 | Associated with lower predicted inactivity risk |
| Monetary | +0.374 | Associated with higher predicted inactivity risk |
| Recency | +0.132 | Associated with higher predicted inactivity risk |
| TenureDays | +0.123 | Associated with higher predicted inactivity risk |
| PurchaseSpanDays | +0.006 | Minimal association |
| AvgOrderValue | âˆ’0.003 | Minimal association |

> **Note:** Monetary, Frequency, TotalItems, and ActiveMonths are likely correlated. Individual
> coefficient magnitudes should not be interpreted as independent causal effects.

---

## Limitations

1. **No native churn label.** The UCI Online Retail dataset does not contain an original churn label.
   `Churn_90D` is a project-defined future-inactivity proxy, not an industry-standard definition.

2. **Proxy label limitations.** A customer recorded as inactive in the 90-day future window may have
   purchased through a different channel or retailer, may have paused spending temporarily, or may
   have genuinely churned. The label cannot distinguish between these cases.

3. **Fixed temporal choices.** The snapshot date (31-Aug-2011), observation period
   (01-Dec-2010 â€“ 31-Aug-2011), and 90-day future window (01-Sep â€“ 29-Nov-2011) are fixed project
   choices. Different window choices would produce different churn rates and model results.

4. **Single retailer and limited time window.** The dataset represents one UK-based online retailer
   over roughly one year (late 2010 â€“ late 2011). Results may not generalise to other retailers,
   markets, or time periods.

5. **Observational data, not experimental.** This project uses observational historical transaction
   data. It does not establish causal effects between any feature and customer churn behaviour.

6. **Probability is a risk-ranking aid.** A predicted churn probability is a relative risk score,
   not a guarantee of future customer behaviour. It should inform prioritisation decisions, not
   replace business judgment.

7. **Retention-priority rules are not validated interventions.** The retention-priority matrix
   encodes intuitive business logic. It has not been validated through controlled experiments or
   A/B testing. Actual retention intervention effectiveness depends on many factors outside this
   model.

---

## Future Scope

The following improvements could extend the system, subject to business requirements and available
data:

- **Extended feature set:** add seasonality indicators, product-category preferences, promotional
  responsiveness, and customer lifetime value estimates.
- **Alternative churn windows:** experiment with 60-day or 120-day future windows.
- **Ensemble methods:** evaluate Random Forest or Gradient Boosting as supplementary comparators
  (while retaining Logistic Regression as the primary interpretable model).
- **Uplift modelling:** move beyond risk ranking to estimate the incremental effect of retention
  interventions for each customer segment.
- **Geographic analysis:** extend to country-level cohort analysis given the multi-country nature
  of the dataset.
- **Recency segmentation:** extend the RFM framework to include win-back segments for lapsed
  customers outside the eligibility window.
- **Model monitoring:** implement drift detection and periodic retraining pipelines for production
  deployment.

---

## References

1. Daqing Chen, Sai Liang Sain, and Kun Guo. *Data mining for the online retail industry: A case
   study of RFM model-based customer segmentation using data mining*. Journal of Database Marketing
   and Customer Strategy Management, 19(3):197â€“208, 2012.
   DOI: [10.1057/dbm.2012.17](https://doi.org/10.1057/dbm.2012.17)

2. UCI Machine Learning Repository â€” Online Retail Dataset.
   <https://archive.ics.uci.edu/ml/datasets/online+retail>

3. Pedregosa et al. *Scikit-learn: Machine Learning in Python*.
   JMLR 12, pp. 2825â€“2830, 2011.

4. Hughes, A. M. *Strategic Database Marketing*. McGraw-Hill, 2005. (RFM methodology.)

5. Streamlit Documentation. <https://docs.streamlit.io>

---

*This project is submitted in partial fulfilment of the IBM SkillsBuild Data Analytics & AI
Internship 2026.*
