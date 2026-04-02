# ChurnGuard - E-Commerce Customer Churn Platform

A full-stack data science platform for predicting and analyzing customer churn in e-commerce, built as part of the ITI Advanced AI Program.

## Features

- Exploratory Data Analysis with automated cleaning & feature engineering
- ML pipeline: Logistic Regression, Random Forest, XGBoost, LightGBM
- SHAP explainability (global + individual)
- MLflow experiment tracking
- FastAPI REST API with prediction, explanation, and what-if endpoints
- Interactive Dash dashboard (8 tabs)
- Power BI executive dashboard
- Real-time streaming simulator & alert engine
- Docker containerization & CI/CD

## Project Structure

```
ecommerce-churn-platform/
├── data/
│   ├── raw/              # Original Kaggle dataset
│   └── processed/        # Cleaned & feature-engineered CSVs
├── notebooks/            # Jupyter notebooks (EDA, modeling)
├── src/
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   └── api/              # FastAPI app
├── dashboards/
│   └── dash_app/         # Dash/Plotly dashboard
├── tests/                # pytest tests
├── models/               # Saved model artifacts
└── .github/workflows/    # CI/CD pipeline
```

## Setup

```bash
# Clone the repo
git clone https://github.com/OmarGamal488/ecommerce-churn-platform.git
cd ecommerce-churn-platform

# Install dependencies
uv sync

# Run Jupyter
uv run jupyter notebook

# Run FastAPI
uv run uvicorn src.api.main:app --port 8000

# Run Dash dashboard
uv run python dashboards/dash_app/app.py
```

## Dataset

[Kaggle: E-Commerce Customer Churn Analysis and Prediction](https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction)

- 5,630 customers x 20 features
- Target: Churn (binary)
- Churn rate: 16.84%

## Team

| Member | Role | Focus |
|--------|------|-------|
| TBD | Data Engineer | EDA, cleaning, ML training |
| TBD | Backend Engineer | FastAPI, streaming, alerts |
| TBD | Frontend Engineer | Dash dashboard |
| TBD | DevOps + Docs | Docker, CI/CD, Power BI, report |
