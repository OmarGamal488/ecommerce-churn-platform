---
license: mit
language:
- en
tags:
- churn-prediction
- e-commerce
- lightgbm
- tabular-classification
- shap
datasets:
- ankitverma2010/ecommerce-customer-churn-analysis-and-prediction
metrics:
- f1
- roc_auc
- accuracy
pipeline_tag: tabular-classification
---

# ChurnGuard — E-Commerce Customer Churn Prediction (LightGBM)

A LightGBM-based model for predicting customer churn in e-commerce, trained on the [Kaggle E-Commerce Churn Dataset](https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction).

## Model Details

- **Model:** LightGBM (via imbalanced-learn pipeline with SMOTE)
- **Task:** Binary classification (Churn: 0 = Stayed, 1 = Churned)
- **Dataset:** 5,630 customers x 30 features (after cleaning & feature engineering)
- **Training:** Stratified 5-fold cross-validation with SMOTE on training folds only

## Performance

| Metric | Score |
|--------|-------|
| **F1 Score** | 0.9574 |
| **AUC-ROC** | 0.9983 |
| **AUC-PR** | 0.9914 |
| **Accuracy** | 0.9858 |

## Files

| File | Description |
|------|-------------|
| `best_model.joblib` | Trained pipeline (SMOTE + LightGBM) |
| `feature_names.joblib` | List of 30 feature column names |
| `shap_explainer.joblib` | SHAP TreeExplainer for model interpretability |

## Usage

```python
import joblib
import pandas as pd

# Load model and feature names
model = joblib.load("best_model.joblib")
feature_names = joblib.load("feature_names.joblib")

# Predict on new data
sample = pd.DataFrame([{
    "Tenure": 4, "CityTier": 3, "WarehouseToHome": 6,
    "Gender": 1, "HourSpendOnApp": 3, "NumberOfDeviceRegistered": 3,
    "SatisfactionScore": 2, "MaritalStatus": 0, "NumberOfAddress": 9,
    "Complain": 1, "OrderAmountHikeFromlastYear": 11,
    "CouponUsed": 1, "OrderCount": 1, "DaySinceLastOrder": 5,
    "CashbackAmount": 159.93, "tenure_bucket": 0,
    "engagement_score": 3, "cashback_per_order": 159.93,
    "is_recent_buyer": 0, "has_multi_device": 0, "is_high_spender": 0,
    "PreferredLoginDevice_Mobile Phone": 1,
    "PreferredPaymentMode_Credit Card": 0,
    "PreferredPaymentMode_Debit Card": 1,
    "PreferredPaymentMode_E wallet": 0,
    "PreferredPaymentMode_UPI": 0,
    "PreferedOrderCat_Grocery": 0,
    "PreferedOrderCat_Laptop & Accessory": 1,
    "PreferedOrderCat_Mobile Phone": 0,
    "PreferedOrderCat_Others": 0,
}])

prediction = model.predict(sample)
probability = model.predict_proba(sample)[:, 1]

print(f"Churn prediction: {prediction[0]}")
print(f"Churn probability: {probability[0]:.4f}")
```

## SHAP Explainability

```python
import shap
import joblib

explainer = joblib.load("shap_explainer.joblib")
shap_values = explainer.shap_values(sample)
shap.waterfall_plot(shap.Explanation(
    values=shap_values[0],
    base_values=explainer.expected_value,
    data=sample.iloc[0],
    feature_names=feature_names,
))
```

## Project

Part of the **ChurnGuard** platform — [GitHub Repository](https://github.com/OmarGamal488/ecommerce-churn-platform)

**Data Exploration Project** | ITI Alexandria | Track AI | INTAKE 46
