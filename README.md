# ChurnGuard — E-Commerce Customer Churn Platform

> A full-stack data science platform for predicting and analyzing customer churn in e-commerce.

**Data Exploration Project** | ITI Alexandria | Track AI | INTAKE 46

[![Live Demo](https://img.shields.io/badge/Live%20Demo-HuggingFace%20Spaces-blue)](https://huggingface.co/spaces/OmarGamal48812/churnguard)
[![Model](https://img.shields.io/badge/Model-HuggingFace%20Hub-yellow)](https://huggingface.co/OmarGamal48812/churn-prediction-lgbm)

---

## Overview

ChurnGuard predicts which e-commerce customers are at risk of churning, explains **why** using SHAP, and recommends **what** the business should do. It combines EDA, machine learning, interactive dashboards, a REST API, and an AI-powered conversational analyst.

### Key Results

| Metric | Score |
|--------|-------|
| **Best Model** | LightGBM |
| **F1 Score** | 0.9574 |
| **AUC-ROC** | 0.9983 |
| **AUC-PR** | 0.9914 |
| **Accuracy** | 0.9858 |

---

## Features

- **EDA & Data Cleaning** — Full exploratory analysis, automated cleaning pipeline, 6 engineered features
- **ML Pipeline** — 4 models (Logistic Regression, Random Forest, XGBoost, LightGBM) with stratified 5-fold CV and SMOTE
- **MLflow Tracking** — All experiments logged with hyperparameters, metrics, and model artifacts
- **SHAP Explainability** — Global feature importance, beeswarm plots, individual waterfall explanations
- **FastAPI REST API** — `/predict`, `/explain`, `/what-if`, `/streaming`, `/alerts/high-risk`
- **Dash Dashboard (6 tabs):**
  - Tab 1: Overview — KPI cards, churn distribution, city tier analysis, gauge
  - Tab 2: Customer Analysis — 4 cross-filters, 6 interactive charts
  - Tab 3: Trend Analysis — Tenure, recency, cashback, coupon patterns
  - Tab 4: Customer Profiles — CRM cards with risk scoring and personalized recommendations
  - Tab 5: ML Predictions — Feature sliders with real-time LightGBM + SHAP
  - Tab 6: AI Analyst — Conversational analytics powered by Gemini + LangChain
- **Power BI Dashboard** — 3-page executive report (Overview, Segmentation, Behavioral Analysis)
- **Docker** — Multi-service containerization (FastAPI + Dash + Streaming)
- **Deployment** — Live on Hugging Face Spaces

---

## Project Structure

```
ecommerce-churn-platform/
├── data/
│   ├── raw/                    # Original Kaggle dataset (E Commerce Dataset.xlsx)
│   └── processed/              # cleaned.csv, engineered.csv
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory Data Analysis
│   ├── 02_cleaning_and_features.ipynb  # Cleaning & feature engineering walkthrough
│   └── 03_modeling.ipynb       # ML training, MLflow, SHAP
├── src/
│   ├── data_cleaning.py        # Reusable clean() function
│   ├── feature_engineering.py  # 6 engineered features + encoding
│   ├── streaming_runner.py     # Background event generator
│   └── api/
│       ├── main.py             # FastAPI app
│       ├── schemas.py          # Pydantic models
│       ├── streaming.py        # Streaming endpoints
│       └── alerts.py           # High-risk alerting
├── dashboards/
│   └── dash_app/
│       ├── app.py              # Main Dash app (6 tabs)
│       ├── data_loader.py      # Shared data loading
│       ├── theme.py            # Color palette & chart config
│       ├── assets/custom.css   # Modern CSS styling
│       └── tabs/               # Tab 1-8 modules
├── models/
│   ├── best_model.joblib       # LightGBM pipeline (SMOTE + classifier)
│   ├── feature_names.joblib    # 30 feature column names
│   └── shap_explainer.joblib   # SHAP TreeExplainer
├── powerbi/
│   ├── project.pbix.zip        # Power BI file
│   └── Power BI Report.docx    # Power BI documentation
├── report/
│   ├── report.tex              # LaTeX source
│   └── report.pdf              # Final report (14 pages)
├── Dockerfile                  # Multi-stage Docker build
├── Dockerfile.spaces           # Hugging Face Spaces deployment
├── docker-compose.yml          # 3-service orchestration
├── start.sh                    # HF Spaces startup script
├── pyproject.toml              # Dependencies (uv)
└── .gitignore
```

---

## Quick Start

### Local Development

```bash
# Clone
git clone https://github.com/OmarGamal488/ecommerce-churn-platform.git
cd ecommerce-churn-platform

# Install dependencies
uv sync

# Run FastAPI
uv run uvicorn src.api.main:app --port 8000

# Run Dash dashboard (in another terminal)
uv run python dashboards/dash_app/app.py
# Open http://localhost:8050
```

### Docker

```bash
docker compose up --build
# Dash: http://localhost:8050
# FastAPI: http://localhost:8000/docs
```

### Notebooks

```bash
uv run jupyter notebook
# Open notebooks/01_eda.ipynb
```

---

## Dataset

**Source:** [Kaggle — E-Commerce Customer Churn Analysis and Prediction](https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction)

- **5,630 customers** × **20 features**
- **Target:** Churn (binary) — 83.16% stayed, 16.84% churned
- **Features:** Tenure, CityTier, PaymentMode, Gender, SatisfactionScore, Complain, CashbackAmount, OrderCount, and more

### Data Quality Issues Found & Fixed

| Issue | Detail | Fix |
|-------|--------|-----|
| Missing values | 1,856 cells (1.65%) across 7 columns | Median imputation |
| Duplicate categories | Phone/Mobile Phone, CC/Credit Card, COD/Cash on Delivery | Merged |
| Outliers | WarehouseToHome: 126, 127 | Corrected to 26, 27 |

---

## Machine Learning

### Models Compared

| Model | F1 | AUC-ROC | AUC-PR | Accuracy |
|-------|-----|---------|--------|----------|
| Logistic Regression | 0.7823 | 0.9241 | 0.8156 | 0.9024 |
| Random Forest | 0.9412 | 0.9954 | 0.9821 | 0.9787 |
| XGBoost | 0.9523 | 0.9976 | 0.9889 | 0.9831 |
| **LightGBM** | **0.9574** | **0.9983** | **0.9914** | **0.9858** |

- **SMOTE** applied inside pipeline (training folds only — no data leakage)
- **Stratified 5-fold CV** for robust evaluation
- **MLflow** tracks all experiments with hyperparameters, metrics, and artifacts
- **SHAP** provides global importance + individual customer explanations

### Feature Engineering

6 new features created from the base dataset:

| Feature | Formula | Rationale |
|---------|---------|-----------|
| tenure_bucket | Bin into new/settling/established/loyal | Non-linear tenure-churn pattern |
| engagement_score | HourSpendOnApp × OrderCount | Combined engagement metric |
| cashback_per_order | CashbackAmount ÷ OrderCount | Normalized monetary signal |
| is_recent_buyer | 1 if DaySinceLastOrder ≤ 3 | Recency flag |
| has_multi_device | 1 if Devices ≥ 4 | Platform investment |
| is_high_spender | 1 if OrderAmountHike > 20% | Spending trajectory |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/model-info` | Model metadata |
| POST | `/predict` | Churn probability + risk level |
| POST | `/explain` | SHAP values per feature |
| POST | `/what-if` | Simulate feature changes |
| POST | `/streaming/generate` | Generate synthetic customer event |
| POST | `/streaming/generate-batch` | Generate multiple events |
| GET | `/streaming/events` | Recent events from queue |
| GET | `/alerts/high-risk` | Customers above churn threshold |
| GET | `/alerts/history` | Alert history log |

API docs: `http://localhost:8000/docs`

---

## Business Questions Answered

| # | Question | Key Finding |
|---|----------|-------------|
| Q1 | Which tenure segments churn most? | First 3 months: 36.2% churn (5× average) |
| Q2 | Does city tier affect churn? | Tier 3: 21.4% vs Tier 1: 14.5% |
| Q3 | Which payment methods drive churn? | Cash on Delivery: 24.9% (highest) |
| Q4 | Which product categories churn? | Mobile Phone: 27.4% (highest + largest segment) |
| Q5 | How does satisfaction relate to churn? | Paradox: high satisfaction ≠ low churn (complaint confound) |
| Q6 | Do complainers churn more? | Yes — 31.7% vs 10.9% (3× multiplier) |
| Q7 | Cashback & order hike impact? | Retained customers get $20 more cashback on average |
| Q8-Q9 | Correlation analysis? | Tenure (-0.35) and Complain (+0.25) are top signals |
| Q10 | RFM profile? | Churners are 1-2 order buyers who never formed habits |

---

## Deployment

### Hugging Face Spaces (Live)

- **Dashboard:** https://huggingface.co/spaces/OmarGamal48812/churnguard
- **Model:** https://huggingface.co/OmarGamal48812/churn-prediction-lgbm

### Architecture

```
Hugging Face Space (Docker)
├── FastAPI (port 8000, background)
│   ├── /predict → LightGBM model
│   ├── /explain → SHAP values
│   └── /alerts → High-risk detection
└── Dash (port 7860, foreground)
    ├── 6 interactive tabs
    └── Calls FastAPI for predictions
```

---

## Team

| Member | Responsibilities |
|--------|-----------------|
| **Ahmed ElSayed** | EDA, data cleaning, feature engineering, ML training (4 models), MLflow tracking, SHAP explainability |
| **Amr Abdel Aziz** | Power BI dashboard (3 pages), DAX measures, Power Query data preparation |
| **Omar Gamal** | Dash dashboard (6 tabs), FastAPI API, AI agent (Gemini + LangChain), Docker, HF deployment, GitHub Projects |

---

## Links

- **Live Dashboard:** https://huggingface.co/spaces/OmarGamal48812/churnguard
- **Model on HF Hub:** https://huggingface.co/OmarGamal48812/churn-prediction-lgbm
- **Dataset:** https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction
