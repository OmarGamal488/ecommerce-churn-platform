"""ChurnGuard — Interactive Dash Dashboard with 8 Tabs."""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

from dashboards.dash_app.tabs.tab1_overview import layout as tab1_layout, register_callbacks as tab1_callbacks
from dashboards.dash_app.tabs.tab2_customer_analysis import layout as tab2_layout, register_callbacks as tab2_callbacks
from dashboards.dash_app.tabs.tab3_trends import layout as tab3_layout, register_callbacks as tab3_callbacks
from dashboards.dash_app.tabs.tab4_profiles import layout as tab4_layout, register_callbacks as tab4_callbacks
from dashboards.dash_app.tabs.tab5_predictions import layout as tab5_layout, register_callbacks as tab5_callbacks
# Tab 6 and 7 deactivated
# from dashboards.dash_app.tabs.tab6_ab_test import layout as tab6_layout, register_callbacks as tab6_callbacks
# from dashboards.dash_app.tabs.tab7_monitoring import layout as tab7_layout, register_callbacks as tab7_callbacks
from dashboards.dash_app.tabs.tab8_ai_analyst import layout as tab8_layout, register_callbacks as tab8_callbacks

GOOGLE_FONTS = "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap"

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY, GOOGLE_FONTS],
    suppress_callback_exceptions=True,
    title="ChurnGuard Dashboard",
)

NAVBAR = dbc.Navbar(
    dbc.Container([
        dbc.NavbarBrand([
            html.Span("Churn", style={"fontWeight": "700"}),
            html.Span("Guard", style={"fontWeight": "400", "opacity": "0.85"}),
        ], className="fs-4 text-white"),
        dbc.Nav([
            dbc.NavItem(dbc.NavLink("Data Exploration Project | ITI Alexandria | Track AI | INTAKE 46",
                                     className="small", style={"color": "rgba(255,255,255,0.7)"})),
        ], className="ms-auto"),
    ], fluid=True),
    dark=True,
    className="mb-4",
)

TABS = dbc.Tabs([
    dbc.Tab(label="Overview", tab_id="tab-1", tab_style={"cursor": "pointer"}),
    dbc.Tab(label="Customer Analysis", tab_id="tab-2", tab_style={"cursor": "pointer"}),
    dbc.Tab(label="Trends", tab_id="tab-3", tab_style={"cursor": "pointer"}),
    dbc.Tab(label="Customer Profiles", tab_id="tab-4", tab_style={"cursor": "pointer"}),
    dbc.Tab(label="Predictions", tab_id="tab-5", tab_style={"cursor": "pointer"}),
    dbc.Tab(label="AI Analyst", tab_id="tab-8", tab_style={"cursor": "pointer"}),
], id="tabs", active_tab="tab-1", className="mb-3")

app.layout = html.Div([
    NAVBAR,
    dbc.Container([
        TABS,
        html.Div(id="tab-content"),
    ], fluid=True),
], style={"minHeight": "100vh", "backgroundColor": "#f4f6f9"})


@app.callback(
    dash.Output("tab-content", "children"),
    dash.Input("tabs", "active_tab"),
)
def render_tab(active_tab):
    tab_map = {
        "tab-1": tab1_layout,
        "tab-2": tab2_layout,
        "tab-3": tab3_layout,
        "tab-4": tab4_layout,
        "tab-5": tab5_layout,
        "tab-8": tab8_layout,
    }
    return tab_map.get(active_tab, html.Div("Tab not found"))


# Register all tab callbacks
tab1_callbacks(app)
tab2_callbacks(app)
tab3_callbacks(app)
tab4_callbacks(app)
tab5_callbacks(app)
# tab6_callbacks(app)
# tab7_callbacks(app)
tab8_callbacks(app)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
