"""High-risk customer alert engine."""

import os
from datetime import datetime
from collections import deque

import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter, Query

from src.api.schemas import HighRiskCustomer, HighRiskResponse
from src.api.streaming import event_queue

router = APIRouter()

# Load model artifacts
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")
_model = joblib.load(os.path.join(MODEL_DIR, "best_model.joblib"))
_feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.joblib"))
_shap_explainer = joblib.load(os.path.join(MODEL_DIR, "shap_explainer.joblib"))

# Alert history
alert_history: deque = deque(maxlen=500)


@router.get("/high-risk", response_model=HighRiskResponse)
def get_high_risk_customers(threshold: float = Query(0.8, ge=0.0, le=1.0)):
    """Scan recent streaming events and return customers with P(churn) > threshold."""
    high_risk = []

    for i, event in enumerate(event_queue):
        customer = event["customer"]
        df = pd.DataFrame([customer])[_feature_names]

        prob = float(_model.predict_proba(df)[0, 1])

        if prob > threshold:
            # Get top risk factors via SHAP
            shap_values = _shap_explainer.shap_values(df)
            impacts = {
                name: round(float(val), 4)
                for name, val in zip(_feature_names, shap_values[0])
            }
            # Top 5 by absolute impact
            top_factors = dict(
                sorted(impacts.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
            )

            entry = HighRiskCustomer(
                customer_index=i,
                churn_probability=round(prob, 4),
                risk_level="Critical" if prob > 0.9 else "High",
                top_risk_factors=top_factors,
            )
            high_risk.append(entry)

            # Log alert
            alert_history.append({
                "timestamp": datetime.now().isoformat(),
                "customer_index": i,
                "churn_probability": round(prob, 4),
                "risk_level": entry.risk_level,
            })

    return HighRiskResponse(
        threshold=threshold,
        count=len(high_risk),
        customers=high_risk,
    )


@router.get("/history")
def get_alert_history(limit: int = Query(50, ge=1, le=500)):
    """Get recent alert history."""
    alerts = list(alert_history)[-limit:]
    return {"count": len(alerts), "alerts": alerts}


@router.delete("/history")
def clear_alert_history():
    """Clear alert history."""
    alert_history.clear()
    return {"status": "cleared"}
