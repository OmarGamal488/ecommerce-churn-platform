"""Tab 5: ML Predictions — feature sliders, real-time prediction, SHAP."""

from dash import html, dcc, Input, Output, State, ALL
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from dashboards.dash_app.data_loader import load_model, load_feature_names, load_shap_explainer, load_engineered
from dashboards.dash_app.theme import *

model = load_model()
feature_names = load_feature_names()
shap_explainer = load_shap_explainer()
df_eng = load_engineered()

# Feature config: name, min, max, step, default
SLIDER_FEATURES = [
    ("Tenure", 0, 61, 1, 10),
    ("CityTier", 1, 3, 1, 1),
    ("WarehouseToHome", 5, 36, 1, 14),
    ("Gender", 0, 1, 1, 0),
    ("HourSpendOnApp", 0, 5, 1, 3),
    ("NumberOfDeviceRegistered", 1, 6, 1, 3),
    ("SatisfactionScore", 1, 5, 1, 3),
    ("MaritalStatus", 0, 2, 1, 0),
    ("NumberOfAddress", 1, 22, 1, 4),
    ("Complain", 0, 1, 1, 0),
    ("OrderAmountHikeFromlastYear", 11, 26, 1, 15),
    ("CouponUsed", 0, 16, 1, 1),
    ("OrderCount", 1, 16, 1, 2),
    ("DaySinceLastOrder", 0, 46, 1, 5),
    ("CashbackAmount", 0, 325, 5, 150),
]

LABEL_MAP = {
    "Gender": {0: "Male", 1: "Female"},
    "MaritalStatus": {0: "Single", 1: "Married", 2: "Divorced"},
    "CityTier": {1: "Tier 1", 2: "Tier 2", 3: "Tier 3"},
    "Complain": {0: "No", 1: "Yes"},
}


def _build_sliders():
    sliders = []
    for name, mn, mx, step, default in SLIDER_FEATURES:
        marks = LABEL_MAP.get(name, {mn: str(mn), mx: str(mx)})
        sliders.append(dbc.Col([
            html.Label(name, className="fw-bold small"),
            dcc.Slider(
                id={"type": "pred-slider", "feature": name},
                min=mn, max=mx, step=step, value=default,
                marks={k: str(v) for k, v in marks.items()} if name in LABEL_MAP else {mn: str(mn), mx: str(mx)},
                tooltip={"placement": "bottom", "always_visible": False},
            ),
        ], md=4, className="mb-3"))
    return sliders


layout = dbc.Container([
    dbc.Row([
        # Left: Sliders
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5("Customer Features", className="mb-3"),
            html.Hr(),
            dbc.Row(_build_sliders()),
            dbc.Button("Predict", id="predict-btn", color="danger", className="w-100 mt-2", size="lg"),
        ]), className="shadow-sm"), md=7),

        # Right: Results
        dbc.Col(
            dbc.Card(dbc.CardBody([
                html.H5("Prediction Result", className="mb-3"),
                dcc.Graph(id="pred-gauge", style={"height": "250px"}),
                html.Div(id="pred-risk-badge", className="text-center"),
            ]), className="shadow-sm"),
        md=5),
    ], className="mb-4"),

    # SHAP Explanation
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5("SHAP Feature Impact — What Drives This Prediction?"),
            dcc.Graph(id="pred-shap-bar"),
        ]), className="shadow-sm"), md=12),
    ]),
], fluid=True)


def _compute_derived(features):
    """Compute derived features from base features."""
    tenure = features["Tenure"]
    if tenure <= 6:
        features["tenure_bucket"] = 0
    elif tenure <= 12:
        features["tenure_bucket"] = 1
    elif tenure <= 24:
        features["tenure_bucket"] = 2
    else:
        features["tenure_bucket"] = 3

    features["engagement_score"] = features["HourSpendOnApp"] * features["OrderCount"]
    features["cashback_per_order"] = (
        features["CashbackAmount"] / features["OrderCount"]
        if features["OrderCount"] > 0 else 0
    )
    features["is_recent_buyer"] = 1 if features["DaySinceLastOrder"] <= 3 else 0
    features["has_multi_device"] = 1 if features["NumberOfDeviceRegistered"] >= 4 else 0
    features["is_high_spender"] = 1 if features["OrderAmountHikeFromlastYear"] > 20 else 0

    # One-hot defaults (all zeros = Cash on Delivery + Computer + Fashion)
    for col in feature_names:
        if col not in features:
            features[col] = 0

    return features


def register_callbacks(app):
    @app.callback(
        Output("pred-gauge", "figure"),
        Output("pred-risk-badge", "children"),
        Output("pred-shap-bar", "figure"),
        Input("predict-btn", "n_clicks"),
        [State({"type": "pred-slider", "feature": name}, "value") for name, *_ in SLIDER_FEATURES],
        prevent_initial_call=True,
    )
    def run_prediction(n_clicks, *slider_values):
        # Build feature dict
        features = {}
        for (name, *_), val in zip(SLIDER_FEATURES, slider_values):
            features[name] = val

        features = _compute_derived(features)
        df_input = pd.DataFrame([features])[feature_names]

        # Predict
        prob = float(model.predict_proba(df_input)[0, 1])
        prediction = int(prob > 0.5)
        risk = "Critical" if prob > 0.8 else "High" if prob > 0.6 else "Medium" if prob > 0.4 else "Low"
        risk_color_map = {"Critical": DANGER, "High": WARNING, "Medium": INFO, "Low": SUCCESS}
        risk_bs = {"Critical": "danger", "High": "warning", "Medium": "info", "Low": "success"}[risk]
        risk_hex = risk_color_map[risk]

        # Gauge
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": CHURNED if prob > 0.5 else STAYED},
                "steps": [
                    {"range": [0, 40], "color": "rgba(46,196,182,0.15)"},
                    {"range": [40, 70], "color": "rgba(255,159,28,0.15)"},
                    {"range": [70, 100], "color": "rgba(231,29,54,0.15)"},
                ],
            },
        ))
        gauge.update_layout(template=PLOTLY_TEMPLATE, margin=dict(t=30, b=10, l=30, r=30), height=220)

        # Risk badge
        badge = html.Div([
            html.H4([
                "Prediction: ",
                dbc.Badge("Churn" if prediction else "Stay", color=risk_bs, className="ms-2 fs-5"),
            ]),
            html.P(f"Risk Level: {risk}", style={"color": risk_hex, "fontWeight": "bold"}),
        ])

        # SHAP
        shap_values = shap_explainer.shap_values(df_input)
        impacts = list(zip(feature_names, shap_values[0]))
        impacts.sort(key=lambda x: abs(x[1]), reverse=True)
        top_n = impacts[:15]

        names = [x[0] for x in top_n][::-1]
        vals = [x[1] for x in top_n][::-1]
        colors = [CHURNED if v > 0 else STAYED for v in vals]

        shap_fig = go.Figure(go.Bar(
            x=vals, y=names, orientation="h",
            marker_color=colors,
            text=[f"{v:+.3f}" for v in vals], textposition="outside",
        ))
        shap_fig.update_layout(
            template=PLOTLY_TEMPLATE,
            margin=dict(t=20, b=20, l=150, r=60),
            height=450,
            xaxis_title="SHAP Value (red = pushes toward churn)",
        )

        return gauge, badge, shap_fig
