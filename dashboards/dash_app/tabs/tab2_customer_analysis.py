"""Tab 2: Customer Analysis & Segmentation — filters, heatmaps, cross-analysis."""

from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from dashboards.dash_app.data_loader import load_raw
from dashboards.dash_app.theme import *

df = load_raw()
# Fix duplicate categories for clean visuals
df["PreferredLoginDevice"] = df["PreferredLoginDevice"].replace("Phone", "Mobile Phone")
df["PreferredPaymentMode"] = df["PreferredPaymentMode"].replace({"CC": "Credit Card", "COD": "Cash on Delivery"})
df["PreferedOrderCat"] = df["PreferedOrderCat"].replace("Mobile", "Mobile Phone")

layout = dbc.Container([
    # Filters Row
    dbc.Row([
        dbc.Col([
            html.Label("City Tier", className="fw-bold"),
            dcc.Dropdown(
                id="filter-city",
                options=[{"label": f"Tier {t}", "value": t} for t in sorted(df["CityTier"].unique())],
                multi=True, placeholder="All Tiers",
            ),
        ], md=3),
        dbc.Col([
            html.Label("Login Device", className="fw-bold"),
            dcc.Dropdown(
                id="filter-device",
                options=[{"label": d, "value": d} for d in df["PreferredLoginDevice"].unique()],
                multi=True, placeholder="All Devices",
            ),
        ], md=3),
        dbc.Col([
            html.Label("Payment Mode", className="fw-bold"),
            dcc.Dropdown(
                id="filter-payment",
                options=[{"label": p, "value": p} for p in df["PreferredPaymentMode"].unique()],
                multi=True, placeholder="All Modes",
            ),
        ], md=3),
        dbc.Col([
            html.Label("Marital Status", className="fw-bold"),
            dcc.Dropdown(
                id="filter-marital",
                options=[{"label": m, "value": m} for m in df["MaritalStatus"].unique()],
                multi=True, placeholder="All",
            ),
        ], md=3),
    ], className="mb-4"),

    # Charts Row 1
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5("Churn Rate by Payment Mode"),
            dcc.Graph(id="ca-payment-bar"),
        ]), className="shadow-sm"), md=6),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5("Churn Rate by Order Category"),
            dcc.Graph(id="ca-category-bar"),
        ]), className="shadow-sm"), md=6),
    ], className="mb-4"),

    # Charts Row 2
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5("Satisfaction × City Tier — Churn Heatmap"),
            dcc.Graph(id="ca-heatmap"),
        ]), className="shadow-sm"), md=6),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5("Complaint Impact on Churn"),
            dcc.Graph(id="ca-complaint"),
        ]), className="shadow-sm"), md=6),
    ], className="mb-4"),

    # Charts Row 3
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5("Churn by Gender & Marital Status"),
            dcc.Graph(id="ca-gender-marital"),
        ]), className="shadow-sm"), md=6),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5("Churn by Login Device"),
            dcc.Graph(id="ca-device-bar"),
        ]), className="shadow-sm"), md=6),
    ]),
], fluid=True)


def register_callbacks(app):
    @app.callback(
        Output("ca-payment-bar", "figure"),
        Output("ca-category-bar", "figure"),
        Output("ca-heatmap", "figure"),
        Output("ca-complaint", "figure"),
        Output("ca-gender-marital", "figure"),
        Output("ca-device-bar", "figure"),
        Input("filter-city", "value"),
        Input("filter-device", "value"),
        Input("filter-payment", "value"),
        Input("filter-marital", "value"),
    )
    def update_analysis(city, device, payment, marital):
        filtered = df.copy()
        if city:
            filtered = filtered[filtered["CityTier"].isin(city)]
        if device:
            filtered = filtered[filtered["PreferredLoginDevice"].isin(device)]
        if payment:
            filtered = filtered[filtered["PreferredPaymentMode"].isin(payment)]
        if marital:
            filtered = filtered[filtered["MaritalStatus"].isin(marital)]

        margins = CHART_MARGINS

        # Payment mode churn rate
        pay_churn = filtered.groupby("PreferredPaymentMode")["Churn"].mean().sort_values(ascending=False) * 100
        fig_pay = go.Figure(go.Bar(
            x=pay_churn.index, y=pay_churn.values,
            marker_color=DANGER,
            text=[f"{v:.1f}%" for v in pay_churn.values], textposition="auto",
        ))
        fig_pay.update_layout(template=PLOTLY_TEMPLATE, margin=margins, height=320, yaxis_title="Churn Rate (%)")

        # Order category churn rate
        cat_churn = filtered.groupby("PreferedOrderCat")["Churn"].mean().sort_values(ascending=False) * 100
        fig_cat = go.Figure(go.Bar(
            x=cat_churn.index, y=cat_churn.values,
            marker_color=WARNING,
            text=[f"{v:.1f}%" for v in cat_churn.values], textposition="auto",
        ))
        fig_cat.update_layout(template=PLOTLY_TEMPLATE, margin=margins, height=320, yaxis_title="Churn Rate (%)")

        # Satisfaction x City heatmap
        pivot = filtered.pivot_table(values="Churn", index="SatisfactionScore",
                                     columns="CityTier", aggfunc="mean") * 100
        fig_heat = go.Figure(go.Heatmap(
            z=pivot.values, x=[f"Tier {c}" for c in pivot.columns],
            y=[str(s) for s in pivot.index],
            colorscale="Blues", text=pivot.values.round(1),
            texttemplate="%{text:.1f}%", colorbar_title="Churn %",
        ))
        fig_heat.update_layout(
            template=PLOTLY_TEMPLATE, margin=margins, height=320,
            xaxis_title="City Tier", yaxis_title="Satisfaction Score",
        )

        # Complaint impact
        comp_churn = filtered.groupby("Complain")["Churn"].mean() * 100
        fig_comp = go.Figure(go.Bar(
            x=["No Complaint", "Complained"],
            y=comp_churn.values,
            marker_color=[STAYED, CHURNED],
            text=[f"{v:.1f}%" for v in comp_churn.values], textposition="auto",
        ))
        fig_comp.update_layout(template=PLOTLY_TEMPLATE, margin=margins, height=320, yaxis_title="Churn Rate (%)")

        # Gender x Marital
        gm = filtered.groupby(["Gender", "MaritalStatus"])["Churn"].mean().reset_index()
        gm["Churn"] = gm["Churn"] * 100
        fig_gm = px.bar(
            gm, x="MaritalStatus", y="Churn", color="Gender",
            barmode="group", text_auto=".1f",
            color_discrete_map={"Male": PRIMARY, "Female": DANGER},
        )
        fig_gm.update_layout(template=PLOTLY_TEMPLATE, margin=margins, height=320, yaxis_title="Churn Rate (%)")

        # Device churn
        dev_churn = filtered.groupby("PreferredLoginDevice")["Churn"].mean().sort_values(ascending=False) * 100
        fig_dev = go.Figure(go.Bar(
            x=dev_churn.index, y=dev_churn.values,
            marker_color=[PRIMARY, SUCCESS],
            text=[f"{v:.1f}%" for v in dev_churn.values], textposition="auto",
        ))
        fig_dev.update_layout(template=PLOTLY_TEMPLATE, margin=margins, height=320, yaxis_title="Churn Rate (%)")

        return fig_pay, fig_cat, fig_heat, fig_comp, fig_gm, fig_dev
