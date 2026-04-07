"""Tab 1: Overview + Live Feed — KPI cards, churn distribution, live event ticker."""

from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import random
from datetime import datetime

from dashboards.dash_app.data_loader import load_raw, load_model, load_feature_names
from dashboards.dash_app.theme import *

df = load_raw()
model = load_model()
feature_names = load_feature_names()

# Pre-compute KPIs
total_customers = len(df)
churn_rate = df["Churn"].mean() * 100
avg_satisfaction = df["SatisfactionScore"].mean()
avg_tenure = df["Tenure"].mean()
total_churned = df["Churn"].sum()
total_stayed = total_customers - total_churned
complaint_rate = df["Complain"].mean() * 100


def _kpi_card(title, value, subtitle="", color="primary"):
    color_map = {"primary": PRIMARY, "info": INFO, "danger": DANGER,
                 "warning": WARNING, "success": SUCCESS}
    hex_color = color_map.get(color, PRIMARY)
    return dbc.Card(
        dbc.CardBody([
            html.H6(title, className="text-muted mb-1"),
            html.H3(value, className="fw-bold mb-0", style={"color": hex_color}),
            html.Small(subtitle, className="text-muted") if subtitle else None,
        ]),
        className=CARD_STYLE,
    )


layout = dbc.Container([
    # KPI Row
    dbc.Row([
        dbc.Col(_kpi_card("Total Customers", f"{total_customers:,}", "in dataset", "info"), md=2),
        dbc.Col(_kpi_card("Churn Rate", f"{churn_rate:.1f}%", f"{total_churned:,} churned", "danger"), md=2),
        dbc.Col(_kpi_card("Avg Satisfaction", f"{avg_satisfaction:.2f}/5", "1=Low, 5=High", "warning"), md=2),
        dbc.Col(_kpi_card("Avg Tenure", f"{avg_tenure:.1f} mo", "months on platform", "success"), md=2),
        dbc.Col(_kpi_card("Complaint Rate", f"{complaint_rate:.1f}%", "filed a complaint", "danger"), md=2),
        dbc.Col(_kpi_card("Stayed", f"{total_stayed:,}", f"{100 - churn_rate:.1f}%", "success"), md=2),
    ], className="mb-4"),

    # Charts Row
    dbc.Row([
        # Churn Distribution
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5("Churn Distribution", className="mb-3"),
            dcc.Graph(id="overview-churn-pie"),
        ]), className="shadow-sm"), md=4),

        # Churn by City Tier
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5("Churn by City Tier", className="mb-3"),
            dcc.Graph(id="overview-city-bar"),
        ]), className="shadow-sm"), md=4),

        # Satisfaction Distribution
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5("Satisfaction Score Distribution", className="mb-3"),
            dcc.Graph(id="overview-satisfaction"),
        ]), className="shadow-sm"), md=4),
    ], className="mb-4"),

    # Gauge Row (full width)
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5("Churn Rate Gauge", className="mb-3"),
            dcc.Graph(id="overview-gauge"),
        ]), className="shadow-sm"), md=4),
    ]),
], fluid=True)


def register_callbacks(app):
    @app.callback(
        Output("overview-churn-pie", "figure"),
        Output("overview-city-bar", "figure"),
        Output("overview-satisfaction", "figure"),
        Output("overview-gauge", "figure"),
        Input("tabs", "active_tab"),
    )
    def update_overview_charts(tab):
        # Churn pie
        churn_counts = df["Churn"].value_counts()
        pie = go.Figure(go.Pie(
            labels=["Stayed", "Churned"],
            values=[churn_counts[0], churn_counts[1]],
            marker_colors=[STAYED, CHURNED],
            hole=0.4,
        ))
        pie.update_layout(template=PLOTLY_TEMPLATE, margin=dict(t=20, b=20, l=20, r=20), height=300)

        # City tier bar
        city_churn = df.groupby("CityTier")["Churn"].mean() * 100
        city_bar = go.Figure(go.Bar(
            x=[f"Tier {t}" for t in city_churn.index],
            y=city_churn.values,
            marker_color=[PRIMARY, WARNING, DANGER],
            text=[f"{v:.1f}%" for v in city_churn.values],
            textposition="auto",
        ))
        city_bar.update_layout(
            template=PLOTLY_TEMPLATE, margin=dict(t=20, b=20, l=20, r=20),
            height=300, yaxis_title="Churn Rate (%)",
        )

        # Satisfaction distribution
        sat = go.Figure(go.Histogram(
            x=df["SatisfactionScore"], nbinsx=5,
            marker_color=WARNING,
        ))
        sat.update_layout(
            template=PLOTLY_TEMPLATE, margin=dict(t=20, b=20, l=20, r=20),
            height=300, xaxis_title="Score", yaxis_title="Count",
        )

        # Gauge
        gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=churn_rate,
            number={"suffix": "%"},
            title={"text": "Overall Churn Rate"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": DANGER},
                "steps": [
                    {"range": [0, 15], "color": "rgba(46,196,182,0.3)"},
                    {"range": [15, 30], "color": "rgba(255,159,28,0.3)"},
                    {"range": [30, 100], "color": "rgba(231,29,54,0.3)"},
                ],
                "threshold": {"line": {"color": "#333", "width": 2}, "value": churn_rate},
            },
        ))
        gauge.update_layout(template=PLOTLY_TEMPLATE, margin=dict(t=60, b=20, l=30, r=30), height=300)

        return pie, city_bar, sat, gauge

