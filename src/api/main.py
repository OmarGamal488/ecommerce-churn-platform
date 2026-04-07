"""FastAPI application for ChurnGuard — churn prediction, explanation, and alerting."""

import os
import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    CustomerFeatures,
    PredictionResponse,
    ExplanationResponse,
    WhatIfRequest,
    WhatIfResponse,
    HighRiskResponse,
    ModelInfoResponse,
)
from src.api.streaming import router as streaming_router
from src.api.alerts import router as alerts_router

# Load model artifacts
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")
model = joblib.load(os.path.join(MODEL_DIR, "best_model.joblib"))
feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.joblib"))
shap_explainer = joblib.load(os.path.join(MODEL_DIR, "shap_explainer.joblib"))

app = FastAPI(
    title="ChurnGuard API",
    description="E-Commerce Customer Churn Prediction & Explainability API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(streaming_router, prefix="/streaming", tags=["Streaming"])
app.include_router(alerts_router, prefix="/alerts", tags=["Alerts"])


def _features_to_df(features: CustomerFeatures) -> pd.DataFrame:
    """Convert CustomerFeatures to a DataFrame matching model input."""
    data = features.model_dump(by_alias=True)
    df = pd.DataFrame([data])
    # Ensure column order matches training
    df = df[feature_names]
    return df


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": model is not None}


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    """Return model metadata."""
    clf = model.named_steps["clf"]
    return ModelInfoResponse(
        model_name="LightGBM",
        model_type=type(clf).__name__,
        n_features=len(feature_names),
        feature_names=feature_names,
        pipeline_steps=[step[0] for step in model.steps],
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(features: CustomerFeatures):
    """Predict churn probability for a customer."""
    df = _features_to_df(features)
    prediction = int(model.predict(df)[0])
    probability = float(model.predict_proba(df)[0, 1])

    return PredictionResponse(
        churn_prediction=prediction,
        churn_probability=round(probability, 4),
        risk_level="High" if probability > 0.7 else "Medium" if probability > 0.4 else "Low",
    )


@app.post("/explain", response_model=ExplanationResponse)
def explain(features: CustomerFeatures):
    """Return SHAP explanation for a customer prediction."""
    df = _features_to_df(features)
    prediction = int(model.predict(df)[0])
    probability = float(model.predict_proba(df)[0, 1])

    shap_values = shap_explainer.shap_values(df)
    feature_impacts = {
        name: round(float(val), 4)
        for name, val in zip(feature_names, shap_values[0])
    }
    # Sort by absolute impact
    feature_impacts = dict(
        sorted(feature_impacts.items(), key=lambda x: abs(x[1]), reverse=True)
    )

    return ExplanationResponse(
        churn_prediction=prediction,
        churn_probability=round(probability, 4),
        base_value=round(float(shap_explainer.expected_value), 4),
        feature_impacts=feature_impacts,
    )


@app.post("/what-if", response_model=WhatIfResponse)
def what_if(request: WhatIfRequest):
    """Simulate feature changes and return new churn probability."""
    # Original prediction
    original_df = _features_to_df(request.original)
    original_prob = float(model.predict_proba(original_df)[0, 1])

    # Apply modifications
    modified_data = request.original.model_dump()
    modified_data.update(request.modifications)
    modified_df = pd.DataFrame([modified_data])[feature_names]
    modified_prob = float(model.predict_proba(modified_df)[0, 1])

    return WhatIfResponse(
        original_probability=round(original_prob, 4),
        modified_probability=round(modified_prob, 4),
        probability_change=round(modified_prob - original_prob, 4),
        modifications=request.modifications,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
