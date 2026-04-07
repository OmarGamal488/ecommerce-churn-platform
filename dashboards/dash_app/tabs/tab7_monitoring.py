"""Tab 7: Model Monitoring — prediction drift, feature drift, alert log."""

from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from dashboards.dash_app.data_loader import load_engineered, load_model, load_feature_names
from dashboards.dash_app.theme import *

df_eng = load_engineered()
model = load_model()
feature_names = load_feature_names()

# Simulate monitoring data over 30 days
np.random.seed(42)
N_DAYS = 30
dates = [datetime.now() - timedelta(days=N_DAYS - i) for i in range(N_DAYS)]

# Simulate daily prediction distributions with slight drift
daily_stats = []
X = df_eng.drop(columns=["Churn"])
base_probas = model.predict_proba(X)[:, 1]
base_mean = base_probas.mean()
base_std = base_probas.std()

for i, dt in enumerate(dates):
    # Add slight drift over time
    drift = 0.002 * i * np.random.uniform(0.5, 1.5)
    day_mean = base_mean + drift + np.random.normal(0, 0.005)
    day_std = base_std + np.random.normal(0, 0.002)
    day_volume = np.random.randint(150, 250)
    day_high_risk = int(day_volume * (0.15 + drift * 2))
    day_alerts = np.random.randint(0, max(1, day_high_risk // 3))

    daily_stats.append({
        "date": dt.strftime("%Y-%m-%d"),
        "mean_probability": round(day_mean, 4),
        "std_probability": round(day_std, 4),
        "prediction_volume": day_volume,
        "high_risk_count": day_high_risk,
        "alerts_triggered": day_alerts,
    })

monitor_df = pd.DataFrame(daily_stats)

# Feature drift: compare first week vs last week distributions
X_values = X.values
first_week = X_values[:len(X_values) // 4]
last_week = X_values[-len(X_values) // 4:]
feature_drift = []
for i, name in enumerate(feature_names):
    mean_diff = abs(float(np.mean(last_week[:, i]) - np.mean(first_week[:, i])))
    feature_drift.append({"feature": name, "drift_score": round(mean_diff, 4)})
drift_df = pd.DataFrame(feature_drift).sort_values("drift_score", ascending=False)

# Simulated alert log
alert_log = []
for i in range(20):
    dt = dates[-(i % N_DAYS + 1)]
    alert_log.append({
        "timestamp": (dt + timedelta(hours=np.random.randint(0, 24))).strftime("%Y-%m-%d %H:%M"),
        "type": np.random.choice(["High-Risk Customer", "Prediction Drift", "Feature Drift", "Volume Spike"]),
        "severity": np.random.choice(["Critical", "Warning", "Info"]),
        "message": np.random.choice([
            "Customer #50234 flagged with 93% churn probability",
            "Mean prediction shifted +2.3% from baseline",
            "Tenure distribution shift detected (KS=0.15)",
            "Prediction volume 40% above daily average",
            "Customer #51002 flagged with 89% churn probability",
            "CashbackAmount distribution drift detected",
        ]),
    })

def _kpi(title, value, color):
    color_map = {"primary": PRIMARY, "info": INFO, "danger": DANGER,
                 "warning": WARNING, "success": SUCCESS}
    hex_color = color_map.get(color, PRIMARY)
    return dbc.Card(dbc.CardBody([
        html.H6(title, className="text-muted mb-1"),
        html.H3(value, className="fw-bold mb-0", style={"color": hex_color}),
    ]), className=CARD_STYLE + " text-center")


layout = dbc.Container([
    # Row 1: Summary KPIs
    dbc.Row([
        dbc.Col(_kpi("Avg Prediction (Today)", f"{monitor_df.iloc[-1]['mean_probability']:.1%}", "info"), md=3),
        dbc.Col(_kpi("High-Risk Today", str(monitor_df.iloc[-1]["high_risk_count"]), "danger"), md=3),
        dbc.Col(_kpi("Alerts (30 days)", str(monitor_df["alerts_triggered"].sum()), "warning"), md=3),
        dbc.Col(_kpi("Total Predictions (30d)", f"{monitor_df['prediction_volume'].sum():,}", "success"), md=3),
    ], className="mb-4"),

    # Row 2: Prediction drift + Volume
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5("Prediction Drift — Mean Churn Probability Over Time"),
            dcc.Graph(id="mon-drift-line"),
        ]), className="shadow-sm"), md=8),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5("Daily Prediction Volume"),
            dcc.Graph(id="mon-volume"),
        ]), className="shadow-sm"), md=4),
    ], className="mb-4"),

    # Row 3: Feature drift + Alert log
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5("Feature Drift Scores (Top 15)"),
            dcc.Graph(id="mon-feature-drift"),
        ]), className="shadow-sm"), md=6),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5("Alert Log"),
            html.Div(id="mon-alert-log", style={"maxHeight": "400px", "overflowY": "auto"}),
        ]), className="shadow-sm"), md=6),
    ]),
], fluid=True)


def register_callbacks(app):
    @app.callback(
        Output("mon-drift-line", "figure"),
        Output("mon-volume", "figure"),
        Output("mon-feature-drift", "figure"),
        Output("mon-alert-log", "children"),
        Input("tabs", "active_tab"),
    )
    def update_monitoring(tab):
        dark = PLOTLY_TEMPLATE
        margins = dict(t=30, b=30, l=40, r=20)

        # Prediction drift line
        fig_drift = go.Figure()
        fig_drift.add_trace(go.Scatter(
            x=monitor_df["date"], y=monitor_df["mean_probability"],
            mode="lines+markers", name="Mean P(churn)",
            line=dict(color=DANGER, width=2),
            fill="tozeroy", fillcolor="rgba(231,29,54,0.08)",
        ))
        # Baseline
        fig_drift.add_hline(
            y=base_mean, line_dash="dash", line_color=WARNING,
            annotation_text=f"Baseline: {base_mean:.1%}",
        )
        fig_drift.update_layout(
            template=dark, margin=margins, height=320,
            xaxis_title="Date", yaxis_title="Mean Churn Probability",
            yaxis_tickformat=".1%",
        )

        # Volume bar
        fig_vol = go.Figure(go.Bar(
            x=monitor_df["date"], y=monitor_df["prediction_volume"],
            marker_color=PRIMARY,
        ))
        fig_vol.update_layout(
            template=dark, margin=margins, height=320,
            xaxis_title="Date", yaxis_title="Predictions",
            xaxis_tickangle=-45,
        )

        # Feature drift horizontal bar
        top_drift = drift_df.head(15)
        fig_feat = go.Figure(go.Bar(
            x=top_drift["drift_score"].values[::-1],
            y=top_drift["feature"].values[::-1],
            orientation="h",
            marker_color=[DANGER if v > 0.5 else WARNING if v > 0.1 else SUCCESS
                          for v in top_drift["drift_score"].values[::-1]],
            text=[f"{v:.3f}" for v in top_drift["drift_score"].values[::-1]],
            textposition="outside",
        ))
        fig_feat.update_layout(
            template=dark, margin=dict(t=20, b=20, l=160, r=60),
            height=400, xaxis_title="Drift Score (|mean diff|)",
        )

        # Alert log table
        severity_color = {"Critical": "danger", "Warning": "warning", "Info": "info"}
        rows = []
        for alert in alert_log:
            rows.append(
                dbc.Alert([
                    dbc.Badge(alert["severity"], color=severity_color.get(alert["severity"], "secondary"),
                              className="me-2"),
                    html.Small(alert["timestamp"], className="text-muted me-2"),
                    html.Span(f"[{alert['type']}] ", className="fw-bold"),
                    html.Span(alert["message"]),
                ], color=severity_color.get(alert["severity"], "secondary"), className="py-2 mb-1")
            )

        return fig_drift, fig_vol, fig_feat, rows
