"""Tab 6: A/B Test Simulator — segment selection, treatment, ROI calculation."""

from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from dashboards.dash_app.data_loader import load_raw, load_engineered, load_model, load_feature_names
from dashboards.dash_app.theme import *

df_raw = load_raw()
df_raw["PreferredLoginDevice"] = df_raw["PreferredLoginDevice"].replace("Phone", "Mobile Phone")
df_raw["PreferredPaymentMode"] = df_raw["PreferredPaymentMode"].replace({"CC": "Credit Card", "COD": "Cash on Delivery"})
df_raw["PreferedOrderCat"] = df_raw["PreferedOrderCat"].replace("Mobile", "Mobile Phone")

df_eng = load_engineered()
model = load_model()
feature_names = load_feature_names()

# Pre-compute churn probabilities
X = df_eng.drop(columns=["Churn"])
df_raw["churn_prob"] = model.predict_proba(X)[:, 1]

layout = dbc.Container([
    dbc.Row([
        # Left: Configuration
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5("A/B Test Configuration", className="mb-3"),
            html.Hr(),

            # Segment selection
            html.Label("Target Segment", className="fw-bold"),
            dcc.Dropdown(
                id="ab-segment",
                options=[
                    {"label": "High-Risk Customers (P > 0.7)", "value": "high_risk"},
                    {"label": "Medium-Risk (0.4 < P < 0.7)", "value": "medium_risk"},
                    {"label": "New Customers (Tenure ≤ 6)", "value": "new"},
                    {"label": "Complainers", "value": "complainers"},
                    {"label": "All Customers", "value": "all"},
                ],
                value="high_risk",
                className="mb-3",
            ),

            # Treatment
            html.Label("Treatment Action", className="fw-bold"),
            dcc.Dropdown(
                id="ab-treatment",
                options=[
                    {"label": "Increase Cashback by 20%", "value": "cashback_20"},
                    {"label": "Increase Cashback by 50%", "value": "cashback_50"},
                    {"label": "Resolve Complaints", "value": "resolve_complaints"},
                    {"label": "Send Coupon (3 extra)", "value": "coupon_3"},
                    {"label": "Personal Follow-up Call", "value": "followup"},
                ],
                value="cashback_20",
                className="mb-3",
            ),

            # Parameters
            html.Label("Average Revenue Per Customer ($)", className="fw-bold"),
            dcc.Input(id="ab-revenue", type="number", value=200, className="form-control mb-3"),

            html.Label("Treatment Cost Per Customer ($)", className="fw-bold"),
            dcc.Input(id="ab-cost", type="number", value=15, className="form-control mb-3"),

            html.Label("Expected Churn Reduction (%)", className="fw-bold"),
            dcc.Slider(id="ab-reduction", min=5, max=50, step=5, value=20,
                       marks={5: "5%", 25: "25%", 50: "50%"},
                       tooltip={"placement": "bottom"}),

            dbc.Button("Run Simulation", id="ab-run-btn", color="primary", className="w-100 mt-3", size="lg"),
        ]), className="shadow-sm"), md=5),

        # Right: Results
        dbc.Col([
            dbc.Card(dbc.CardBody([
                html.H5("Simulation Results", className="mb-3"),
                html.Div(id="ab-results"),
            ]), className="shadow-sm mb-3"),

            dbc.Card(dbc.CardBody([
                html.H5("ROI Comparison", className="mb-3"),
                dcc.Graph(id="ab-roi-chart"),
            ]), className="shadow-sm mb-3"),

            dbc.Card(dbc.CardBody([
                html.H5("Churn Rate — Control vs Treatment", className="mb-3"),
                dcc.Graph(id="ab-churn-comparison"),
            ]), className="shadow-sm"),
        ], md=7),
    ]),
], fluid=True)


def register_callbacks(app):
    @app.callback(
        Output("ab-results", "children"),
        Output("ab-roi-chart", "figure"),
        Output("ab-churn-comparison", "figure"),
        Input("ab-run-btn", "n_clicks"),
        State("ab-segment", "value"),
        State("ab-treatment", "value"),
        State("ab-revenue", "value"),
        State("ab-cost", "value"),
        State("ab-reduction", "value"),
        prevent_initial_call=True,
    )
    def run_ab_simulation(n_clicks, segment, treatment, revenue, cost, reduction_pct):
        # Select segment
        if segment == "high_risk":
            mask = df_raw["churn_prob"] > 0.7
        elif segment == "medium_risk":
            mask = (df_raw["churn_prob"] > 0.4) & (df_raw["churn_prob"] <= 0.7)
        elif segment == "new":
            mask = df_raw["Tenure"] <= 6
        elif segment == "complainers":
            mask = df_raw["Complain"] == 1
        else:
            mask = pd.Series(True, index=df_raw.index)

        segment_df = df_raw[mask]
        n_customers = len(segment_df)

        if n_customers == 0:
            return dbc.Alert("No customers in this segment.", color="warning"), go.Figure(), go.Figure()

        # Control group stats
        control_churn_rate = segment_df["Churn"].mean()
        control_churned = int(control_churn_rate * n_customers)
        control_retained = n_customers - control_churned

        # Treatment group (simulated)
        reduction = reduction_pct / 100
        treatment_churn_rate = control_churn_rate * (1 - reduction)
        treatment_churned = int(treatment_churn_rate * n_customers)
        treatment_retained = n_customers - treatment_churned

        # ROI calculation
        customers_saved = control_churned - treatment_churned
        revenue_saved = customers_saved * revenue
        total_cost = n_customers * cost
        net_roi = revenue_saved - total_cost
        roi_pct = (net_roi / total_cost * 100) if total_cost > 0 else 0

        # Results KPIs
        results = dbc.Container([
            dbc.Row([
                dbc.Col(_kpi("Segment Size", f"{n_customers:,}", "info"), md=4),
                dbc.Col(_kpi("Customers Saved", f"{customers_saved:,}", "success"), md=4),
                dbc.Col(_kpi("Net ROI", f"${net_roi:,.0f}", "success" if net_roi > 0 else "danger"), md=4),
            ], className="mb-3"),
            dbc.Row([
                dbc.Col(_kpi("Revenue Saved", f"${revenue_saved:,.0f}", "success"), md=4),
                dbc.Col(_kpi("Treatment Cost", f"${total_cost:,.0f}", "warning"), md=4),
                dbc.Col(_kpi("ROI %", f"{roi_pct:.1f}%", "success" if roi_pct > 0 else "danger"), md=4),
            ]),
        ], fluid=True)

        # ROI bar chart
        roi_fig = go.Figure()
        roi_fig.add_trace(go.Bar(
            x=["Revenue Saved", "Treatment Cost", "Net ROI"],
            y=[revenue_saved, total_cost, net_roi],
            marker_color=[SUCCESS, WARNING, PRIMARY if net_roi > 0 else DANGER],
            text=[f"${v:,.0f}" for v in [revenue_saved, total_cost, net_roi]],
            textposition="auto",
        ))
        roi_fig.update_layout(
            template=PLOTLY_TEMPLATE, margin=dict(t=20, b=30, l=40, r=20),
            height=280, yaxis_title="Amount ($)",
        )

        # Churn comparison
        churn_fig = go.Figure()
        churn_fig.add_trace(go.Bar(
            name="Control", x=["Churned", "Retained"],
            y=[control_churned, control_retained],
            marker_color=[CHURNED, STAYED],
            text=[f"{control_churned}", f"{control_retained}"], textposition="auto",
        ))
        churn_fig.add_trace(go.Bar(
            name="Treatment", x=["Churned", "Retained"],
            y=[treatment_churned, treatment_retained],
            marker_color=["#c0392b", "#239e8e"],
            text=[f"{treatment_churned}", f"{treatment_retained}"], textposition="auto",
        ))
        churn_fig.update_layout(
            template=PLOTLY_TEMPLATE, margin=dict(t=20, b=30, l=40, r=20),
            height=280, barmode="group", yaxis_title="Customers",
        )

        return results, roi_fig, churn_fig


def _kpi(title, value, color):
    color_map = {"primary": PRIMARY, "info": INFO, "danger": DANGER,
                 "warning": WARNING, "success": SUCCESS}
    hex_color = color_map.get(color, PRIMARY)
    return dbc.Card(dbc.CardBody([
        html.H6(title, className="text-muted mb-1 small"),
        html.H4(value, className="fw-bold mb-0", style={"color": hex_color}),
    ]), className="text-center " + CARD_STYLE)
