"""Tab 4: Customer Profiles — CRM-style cards with risk scoring and recommendations."""

from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np

from dashboards.dash_app.data_loader import load_raw, load_model, load_engineered, load_feature_names
from dashboards.dash_app.theme import *

df_raw = load_raw()
df_eng = load_engineered()
model = load_model()
feature_names = load_feature_names()

# Pre-compute predictions for all customers
X = df_eng.drop(columns=["Churn"])
probas = model.predict_proba(X)[:, 1]
df_raw["churn_prob"] = probas
df_raw["risk_level"] = pd.cut(probas, bins=[0, 0.3, 0.6, 0.8, 1.0],
                               labels=["Low", "Medium", "High", "Critical"])

layout = dbc.Container([
    # Search & Filter
    dbc.Row([
        dbc.Col([
            html.Label("Search by Customer ID", className="fw-bold"),
            dcc.Input(id="profile-search", type="number", placeholder="e.g. 50001",
                      className="form-control", min=50001, max=55630),
        ], md=3),
        dbc.Col([
            html.Label("Filter by Risk Level", className="fw-bold"),
            dcc.Dropdown(
                id="profile-risk-filter",
                options=[
                    {"label": "All", "value": "all"},
                    {"label": "Critical", "value": "Critical"},
                    {"label": "High", "value": "High"},
                    {"label": "Medium", "value": "Medium"},
                    {"label": "Low", "value": "Low"},
                ],
                value="all",
            ),
        ], md=3),
        dbc.Col([
            html.Label("Sort By", className="fw-bold"),
            dcc.Dropdown(
                id="profile-sort",
                options=[
                    {"label": "Highest Risk First", "value": "desc"},
                    {"label": "Lowest Risk First", "value": "asc"},
                ],
                value="desc",
            ),
        ], md=3),
        dbc.Col([
            html.Label("Page", className="fw-bold"),
            dcc.Input(id="profile-page", type="number", value=1, min=1,
                      className="form-control"),
        ], md=3),
    ], className="mb-4"),

    # Customer Cards
    html.Div(id="profile-cards"),

    # Pagination info
    html.Div(id="profile-pagination", className="text-muted text-center mt-3"),
], fluid=True)


def _risk_color(risk):
    return {"Critical": "danger", "High": "warning", "Medium": "info", "Low": "success"}.get(risk, "secondary")


def _recommendation(row):
    recs = []
    if row.get("Complain") == 1:
        recs.append("Resolve complaint urgently — complaints strongly predict churn")
    if row.get("churn_prob", 0) > 0.7:
        recs.append("Offer personalized retention incentive (discount/cashback)")
    if row.get("Tenure", 99) <= 6:
        recs.append("New customer — onboarding follow-up recommended")
    if row.get("DaySinceLastOrder", 0) and row["DaySinceLastOrder"] > 15:
        recs.append("Inactive for 15+ days — send re-engagement campaign")
    if row.get("SatisfactionScore", 5) <= 2:
        recs.append("Low satisfaction — schedule feedback call")
    if not recs:
        recs.append("Customer appears healthy — maintain current engagement")
    return recs


def _customer_card(row):
    risk = row.get("risk_level", "Unknown")
    prob = row.get("churn_prob", 0)
    recs = _recommendation(row)

    return dbc.Card(dbc.CardBody([
        dbc.Row([
            # Left: Customer info
            dbc.Col([
                html.H5([
                    f"Customer #{int(row['CustomerID'])} ",
                    dbc.Badge(risk, color=_risk_color(risk), className="ms-2"),
                ]),
                html.Hr(className="my-2"),
                dbc.Row([
                    dbc.Col([
                        html.Small("Gender", className="text-muted d-block"),
                        html.Strong(row.get("Gender", "N/A")),
                    ], md=3),
                    dbc.Col([
                        html.Small("Marital Status", className="text-muted d-block"),
                        html.Strong(str(row.get("MaritalStatus", "N/A"))),
                    ], md=3),
                    dbc.Col([
                        html.Small("City Tier", className="text-muted d-block"),
                        html.Strong(f"Tier {row.get('CityTier', 'N/A')}"),
                    ], md=3),
                    dbc.Col([
                        html.Small("Tenure", className="text-muted d-block"),
                        html.Strong(f"{row.get('Tenure', 'N/A')} mo"),
                    ], md=3),
                ], className="mb-2"),
                dbc.Row([
                    dbc.Col([
                        html.Small("Satisfaction", className="text-muted d-block"),
                        html.Strong(f"{row.get('SatisfactionScore', 'N/A')}/5"),
                    ], md=3),
                    dbc.Col([
                        html.Small("Orders", className="text-muted d-block"),
                        html.Strong(str(row.get("OrderCount", "N/A"))),
                    ], md=3),
                    dbc.Col([
                        html.Small("Cashback", className="text-muted d-block"),
                        html.Strong(f"${row.get('CashbackAmount', 0):.0f}"),
                    ], md=3),
                    dbc.Col([
                        html.Small("Complaint", className="text-muted d-block"),
                        html.Strong("Yes" if row.get("Complain") == 1 else "No",
                                    className="text-danger" if row.get("Complain") == 1 else ""),
                    ], md=3),
                ]),
            ], md=7),

            # Right: Risk & Recommendations
            dbc.Col([
                html.Div([
                    html.H6("Churn Probability", className="text-muted mb-1"),
                    html.H2(f"{prob:.0%}", className=f"text-{_risk_color(risk)} fw-bold"),
                ], className="text-center mb-3"),
                html.H6("Recommendations", className="text-muted"),
                html.Ul([html.Li(r, className="small") for r in recs]),
            ], md=5),
        ]),
    ]), className="shadow-sm mb-3")


def register_callbacks(app):
    @app.callback(
        Output("profile-cards", "children"),
        Output("profile-pagination", "children"),
        Input("profile-search", "value"),
        Input("profile-risk-filter", "value"),
        Input("profile-sort", "value"),
        Input("profile-page", "value"),
    )
    def update_profiles(search_id, risk_filter, sort_order, page):
        PAGE_SIZE = 6
        filtered = df_raw.copy()

        # Search by ID
        if search_id:
            filtered = filtered[filtered["CustomerID"] == search_id]
            if filtered.empty:
                return html.Div(
                    dbc.Alert(f"Customer #{search_id} not found.", color="warning"),
                ), ""

        # Filter by risk
        if risk_filter and risk_filter != "all":
            filtered = filtered[filtered["risk_level"] == risk_filter]

        # Sort
        ascending = sort_order == "asc"
        filtered = filtered.sort_values("churn_prob", ascending=ascending)

        # Paginate
        page = max(1, page or 1)
        total = len(filtered)
        total_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
        start = (page - 1) * PAGE_SIZE
        end = start + PAGE_SIZE
        page_data = filtered.iloc[start:end]

        cards = [_customer_card(row) for _, row in page_data.iterrows()]
        pagination = f"Showing {start+1}-{min(end, total)} of {total} customers (Page {page}/{total_pages})"

        return cards, pagination
