"""Tab 3: Trend Analysis — tenure, recency, cashback, coupon usage vs churn."""

from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from dashboards.dash_app.data_loader import load_raw
from dashboards.dash_app.theme import *

df = load_raw()
df["PreferredLoginDevice"] = df["PreferredLoginDevice"].replace("Phone", "Mobile Phone")
df["PreferedOrderCat"] = df["PreferedOrderCat"].replace("Mobile", "Mobile Phone")

layout = dbc.Container([
    # Row 1: Tenure
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5("Tenure Distribution by Churn"),
            dcc.Graph(id="trend-tenure-hist"),
        ]), className="shadow-sm"), md=6),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5("Churn Rate by Tenure Bucket"),
            dcc.Graph(id="trend-tenure-bucket"),
        ]), className="shadow-sm"), md=6),
    ], className="mb-4"),

    # Row 2: Recency & Orders
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5("Days Since Last Order vs Churn"),
            dcc.Graph(id="trend-recency"),
        ]), className="shadow-sm"), md=6),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5("Order Count vs Churn"),
            dcc.Graph(id="trend-orders"),
        ]), className="shadow-sm"), md=6),
    ], className="mb-4"),

    # Row 3: Cashback & Coupons
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5("Cashback Amount vs Churn Probability"),
            dcc.Graph(id="trend-cashback"),
        ]), className="shadow-sm"), md=6),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5("Coupon Usage vs Churn"),
            dcc.Graph(id="trend-coupon"),
        ]), className="shadow-sm"), md=6),
    ]),
], fluid=True)


def register_callbacks(app):
    @app.callback(
        Output("trend-tenure-hist", "figure"),
        Output("trend-tenure-bucket", "figure"),
        Output("trend-recency", "figure"),
        Output("trend-orders", "figure"),
        Output("trend-cashback", "figure"),
        Output("trend-coupon", "figure"),
        Input("tabs", "active_tab"),
    )
    def update_trends(tab):
        margins = CHART_MARGINS

        # Tenure histogram by churn
        stayed = df[df["Churn"] == 0]["Tenure"].dropna()
        churned = df[df["Churn"] == 1]["Tenure"].dropna()
        fig_tenure = go.Figure()
        fig_tenure.add_trace(go.Histogram(x=stayed, name="Stayed", marker_color=STAYED, opacity=0.7, nbinsx=20))
        fig_tenure.add_trace(go.Histogram(x=churned, name="Churned", marker_color=CHURNED, opacity=0.7, nbinsx=20))
        fig_tenure.update_layout(
            template=PLOTLY_TEMPLATE, margin=margins, height=320,
            barmode="overlay", xaxis_title="Tenure (months)", yaxis_title="Count",
        )

        # Tenure bucket churn rate
        bins = [0, 6, 12, 24, 61]
        labels = ["0-6 (New)", "7-12 (Settling)", "13-24 (Established)", "25+ (Loyal)"]
        df["_bucket"] = pd.cut(df["Tenure"], bins=bins, labels=labels, include_lowest=True)
        bucket_churn = df.groupby("_bucket", observed=True)["Churn"].mean() * 100
        fig_bucket = go.Figure(go.Bar(
            x=bucket_churn.index.astype(str), y=bucket_churn.values,
            marker_color=[DANGER, WARNING, INFO, SUCCESS],
            text=[f"{v:.1f}%" for v in bucket_churn.values], textposition="auto",
        ))
        fig_bucket.update_layout(template=PLOTLY_TEMPLATE, margin=margins, height=320, yaxis_title="Churn Rate (%)")
        df.drop(columns="_bucket", inplace=True)

        # Recency — box plot
        fig_recency = go.Figure()
        fig_recency.add_trace(go.Box(
            y=df[df["Churn"] == 0]["DaySinceLastOrder"].dropna(),
            name="Stayed", marker_color=STAYED,
        ))
        fig_recency.add_trace(go.Box(
            y=df[df["Churn"] == 1]["DaySinceLastOrder"].dropna(),
            name="Churned", marker_color=CHURNED,
        ))
        fig_recency.update_layout(template=PLOTLY_TEMPLATE, margin=margins, height=320, yaxis_title="Days Since Last Order")

        # Order count — box plot
        fig_orders = go.Figure()
        fig_orders.add_trace(go.Box(
            y=df[df["Churn"] == 0]["OrderCount"].dropna(),
            name="Stayed", marker_color=STAYED,
        ))
        fig_orders.add_trace(go.Box(
            y=df[df["Churn"] == 1]["OrderCount"].dropna(),
            name="Churned", marker_color=CHURNED,
        ))
        fig_orders.update_layout(template=PLOTLY_TEMPLATE, margin=margins, height=320, yaxis_title="Order Count")

        # Cashback scatter
        sample = df.dropna(subset=["CashbackAmount", "Tenure"]).sample(min(1000, len(df)), random_state=42)
        fig_cash = px.scatter(
            sample, x="CashbackAmount", y="Tenure", color="Churn",
            color_discrete_map={0: STAYED, 1: CHURNED},
            opacity=0.5, labels={"Churn": "Churn"},
        )
        fig_cash.update_layout(template=PLOTLY_TEMPLATE, margin=margins, height=320)

        # Coupon usage — bar chart
        coupon_churn = df.groupby(
            pd.cut(df["CouponUsed"].dropna(), bins=[0, 1, 3, 5, 16],
                   labels=["0-1", "2-3", "4-5", "6+"], include_lowest=True),
            observed=True,
        )["Churn"].mean() * 100
        fig_coupon = go.Figure(go.Bar(
            x=coupon_churn.index.astype(str), y=coupon_churn.values,
            marker_color=PRIMARY,
            text=[f"{v:.1f}%" for v in coupon_churn.values], textposition="auto",
        ))
        fig_coupon.update_layout(
            template=PLOTLY_TEMPLATE, margin=margins, height=320,
            xaxis_title="Coupons Used", yaxis_title="Churn Rate (%)",
        )

        return fig_tenure, fig_bucket, fig_recency, fig_orders, fig_cash, fig_coupon
