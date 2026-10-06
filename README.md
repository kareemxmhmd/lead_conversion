# Bank Marketing Lead Conversion & Scoring System

A production-grade machine learning system designed to predict customer conversion propensity for bank term deposit telemarketing campaigns (`bank.csv`).

---

## 1. Problem & Business Context

Direct marketing campaigns conducted via human telemarketing calls are costly, constrained by agent bandwidth, and risk customer fatigue. Calling every lead uniformly results in low conversion rates (~11.5%) and wasted sales capacity.

**Business Objective:**
Rank and qualify prospective bank leads prior to outbound dialing, enabling sales teams to prioritize high-conversion prospects and achieve over **3.2x conversion lift in the top decile**.

---

## 2. Dataset Overview

- **Source**: UCI Bank Marketing Dataset (Moro et al., 2014)
- **Records**: 4,521 customers
- **Features**: 16 raw client, financial, and campaign attributes
- **Target (`y`)**: Subscribed to a term deposit (`no`: 4,000 / 88.48%, `yes`: 521 / 11.52%)

### The `duration` Dilemma & Leakage Prevention
> Call duration (`duration`) is only known **after** a call finishes and strongly correlates with interest. In a real-world pre-call lead scoring system, call duration is unknown ($0$). 
> To guarantee a **realistic, leakage-free operational model**, the operational pre-call lead scoring pipeline excludes `duration` at inference time.

---

## 3. Machine Learning Architecture

```
DATA (bank.csv)
  │
  ▼
Stratified Train/Val/Test Split (70% / 15% / 15%)
  │
  ▼
Feature Engineering (Contact history indicators, overdraft flags, loan burden)
  │
  ▼
Scikit-Learn ColumnTransformer (StandardScaler + OneHotEncoder)
  │
  ▼
Model Benchmark (Dummy Baseline, Logistic Regression, Random Forest, LightGBM, XGBoost)
  │
  ▼
Champion Selection & Threshold Tuning (tau* = 0.69)
  │
  ▼
Serialized Artifacts (model.pkl, preprocessor.pkl, features.json, metadata.json)
  │
  ▼
Production FastAPI Backend (Pydantic v2, REST Endpoints, Batch Scoring)
```

---

## 4. Model Evaluation & Benchmark Results

### Validation Comparison
| Model | 5-Fold CV PR-AUC | Val PR-AUC | Val ROC-AUC | F1 ($\tau^*$) | Top 10% Decile Lift |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Dummy (Majority)** | 0.1154 $\pm$ 0.0001 | 0.1150 | 0.5000 | 0.0000 | 0.52x |
| **Logistic Regression (Balanced)** | 0.3173 $\pm$ 0.0324 | 0.4197 | 0.7110 | 0.4457 | 4.02x |
| **Random Forest (Balanced)** | 0.3231 $\pm$ 0.0225 | 0.4046 | 0.7191 | 0.4580 | 3.89x |
| **LightGBM (Weighted) [Champion]** | **0.3177 $\pm$ 0.0069** | **0.4304** | **0.7245** | **0.4697** | **4.28x** |
| **XGBoost (Weighted)** | 0.3146 $\pm$ 0.0277 | 0.3727 | 0.6921 | 0.4384 | 4.15x |

### Champion Model Unseen Test Set Performance
- **Champion Algorithm**: `LightGBM (Weighted)`
- **Decision Threshold ($\tau^*$)**: `0.69`
- **Test PR-AUC**: **0.3219** (95% Bootstrap CI: `[0.2221, 0.4335]`)
- **Test ROC-AUC**: **0.7188** (95% Bootstrap CI: `[0.6468, 0.7815]`)
- **Test Top 10% Decile Lift**: **3.25x** (Captures **32.0%** of all converting customers)
- **Test Top 20% Decile Lift**: **2.45x** (Captures **48.7%** of all converting customers)
- **Test Brier Score**: `0.1505`

---

## 5. Project Structure

```
lead_conversion/
├── api/
│   ├── __init__.py
│   └── main.py                  # Production FastAPI service with Pydantic v2
├── artifacts/
│   ├── features.json            # Feature manifest and column types
│   └── preprocessor.pkl         # Fitted scikit-learn preprocessing pipeline
├── data/
│   ├── bank.csv                 # Raw dataset
│   └── processed/               # Saved stratified splits (train/val/test)
├── models/
│   ├── metadata.json            # Model card & training hyperparams
│   ├── metrics.json             # PR-AUC, ROC-AUC, F1, Decile Lift across splits
│   └── model.pkl                # Selected champion model
├── src/
│   ├── __init__.py
│   ├── config.py                # Paths, constants, and hyperparameters
│   ├── data_loader.py           # Semicolon CSV loader & data validation
│   ├── preprocessing.py         # Sklearn ColumnTransformer & encodings
│   ├── feature_engineering.py   # Domain features (contact indicators, balance flags)
│   ├── train_model.py           # Multi-model training and cross-validation
│   ├── evaluate.py              # Metrics, lift calculation, bootstrap CIs
│   └── select_best.py           # Champion model selection, threshold tuning & export
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py      # Schema and data loading tests
│   ├── test_pipeline.py         # Preprocessing and feature engineering unit tests
│   └── test_api.py              # FastAPI endpoint integration tests
├── .gitignore                   # Python/ML artifact exclusions
├── requirements.txt             # Pinned project dependencies
└── README.md                    # Project documentation
```

---

## 6. Quickstart & Local Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Data Processing & Preprocessing
```bash
python -m src.preprocessing
```

### 3. Train Models & Select Champion
```bash
python -m src.select_best
```

### 4. Run Automated Test Suite
```bash
pytest -v
```

### 5. Launch FastAPI Backend
```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```
Interactive Swagger documentation is available at: `http://127.0.0.1:8000/docs`

---

## 7. API Reference

### Health Check
`GET /health`
```json
{
  "status": "healthy",
  "model_loaded": true,
  "champion_model": "LightGBM (Weighted)",
  "timestamp_utc": "2026-10-06T15:10:00Z"
}
```

### Single Lead Scoring
`POST /predict`

**Request:**
```json
{
  "age": 42,
  "job": "management",
  "marital": "married",
  "education": "tertiary",
  "default": "no",
  "balance": 8500,
  "housing": "no",
  "loan": "no",
  "contact": "cellular",
  "day": 15,
  "month": "oct",
  "campaign": 1,
  "pdays": 90,
  "previous": 2,
  "poutcome": "success"
}
```

**Response:**
```json
{
  "conversion_probability": 0.8124,
  "will_convert": true,
  "lead_tier": "High",
  "decision_threshold": 0.69,
  "recommendation": "High Priority: Contact immediately with premium deposit offering."
}
```

### Batch Lead Scoring
`POST /predict/batch`

Accepts a list of `leads` and returns individual predictions along with aggregate tier distribution counts (`total_leads`, `high_tier_count`, `medium_tier_count`, `low_tier_count`).
