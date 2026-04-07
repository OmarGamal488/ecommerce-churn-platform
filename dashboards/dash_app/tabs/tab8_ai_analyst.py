"""Tab 8: AI Churn Analyst — conversational analytics powered by LangChain + Gemini."""

import os
import requests
import pandas as pd
from dotenv import load_dotenv

from dash import html, dcc, Input, Output, State, no_update, callback_context
import dash_bootstrap_components as dbc

from dashboards.dash_app.data_loader import load_cleaned

# ---------------------------------------------------------------------------
# Module-level setup
# ---------------------------------------------------------------------------
load_dotenv()

df = load_cleaned()

GEMINI_AVAILABLE = bool(os.environ.get("GOOGLE_API_KEY"))
FASTAPI_URL = os.environ.get("FASTAPI_URL", "http://localhost:8000")

agent_executor = None

if GEMINI_AVAILABLE:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langgraph.prebuilt import create_react_agent
    from langchain_core.tools import tool

    # -----------------------------------------------------------------------
    # Tool 1: query_dataframe
    # -----------------------------------------------------------------------
    @tool
    def query_dataframe(python_code: str) -> str:
        """Execute pandas code on the e-commerce churn dataset and return the result.

        The DataFrame is named `df` and has these columns:
        - Churn (int 0/1): 0=stayed, 1=churned. 16.84% churn rate.
        - Tenure (float): Months on the platform (0-61)
        - PreferredLoginDevice (str): "Mobile Phone" or "Computer"
        - CityTier (int): 1, 2, or 3
        - WarehouseToHome (float): Distance in km (5-36)
        - PreferredPaymentMode (str): "Debit Card", "Credit Card", "E wallet", "UPI", "Cash on Delivery"
        - Gender (str): "Male" or "Female"
        - HourSpendOnApp (float): Hours on app (0-5)
        - NumberOfDeviceRegistered (int): Devices registered (1-6)
        - PreferedOrderCat (str): "Laptop & Accessory", "Mobile Phone", "Fashion", "Grocery", "Others"
        - SatisfactionScore (int): 1-5
        - MaritalStatus (str): "Single", "Married", or "Divorced"
        - NumberOfAddress (int): 1-22
        - Complain (int): 0=No, 1=Yes
        - OrderAmountHikeFromlastYear (float): % hike (11-26)
        - CouponUsed (float): Coupons used (0-16)
        - OrderCount (float): Orders placed (1-16)
        - DaySinceLastOrder (float): Days since last order (0-46)
        - CashbackAmount (float): Cashback received (0-325)

        There are 5,630 rows total.

        Args:
            python_code: Valid pandas expression using `df`. Must produce a result
                         via the last expression (do NOT use print).
                         Example: df.groupby("MaritalStatus")["Churn"].mean()
        """
        try:
            local_vars = {"df": df, "pd": pd}
            exec_result = eval(python_code, {"__builtins__": {}}, local_vars)
            result_str = str(exec_result)
            if len(result_str) > 3000:
                result_str = result_str[:3000] + "\n... (truncated)"
            return result_str
        except Exception as e:
            return f"Error executing code: {e}"

    # -----------------------------------------------------------------------
    # Tool 2: predict_churn
    # -----------------------------------------------------------------------
    @tool
    def predict_churn(
        tenure: float,
        city_tier: int,
        warehouse_to_home: float,
        gender: str,
        hour_spend_on_app: float,
        number_of_device_registered: int,
        satisfaction_score: int,
        marital_status: str,
        number_of_address: int,
        complain: int,
        order_amount_hike: float,
        coupon_used: float,
        order_count: float,
        day_since_last_order: float,
        cashback_amount: float,
    ) -> str:
        """Predict churn probability for a customer by calling the ML model API.

        Use reasonable defaults for any features the user does not specify:
        tenure=10, city_tier=1, warehouse_to_home=14, gender="Male",
        hour_spend_on_app=3, number_of_device_registered=3, satisfaction_score=3,
        marital_status="Single", number_of_address=4, complain=0,
        order_amount_hike=15, coupon_used=1, order_count=2,
        day_since_last_order=5, cashback_amount=150.

        Args:
            tenure: Months on the platform (0-61)
            city_tier: City tier (1, 2, or 3)
            warehouse_to_home: Distance in km (5-36)
            gender: "Male" or "Female"
            hour_spend_on_app: Hours on app (0-5)
            number_of_device_registered: Devices registered (1-6)
            satisfaction_score: Score 1-5
            marital_status: "Single", "Married", or "Divorced"
            number_of_address: Number of addresses (1-22)
            complain: 0=No, 1=Yes
            order_amount_hike: Order amount hike from last year % (11-26)
            coupon_used: Coupons used (0-16)
            order_count: Orders placed (1-16)
            day_since_last_order: Days since last order (0-46)
            cashback_amount: Cashback received (0-325)
        """
        gender_val = 1 if gender.lower() == "female" else 0
        marital_map = {"single": 0, "married": 1, "divorced": 2}
        marital_val = marital_map.get(marital_status.lower(), 0)

        tenure_bucket = 0 if tenure <= 6 else 1 if tenure <= 12 else 2 if tenure <= 24 else 3
        engagement_score = hour_spend_on_app * order_count
        cashback_per_order = cashback_amount / order_count if order_count > 0 else 0
        is_recent_buyer = 1 if day_since_last_order <= 3 else 0
        has_multi_device = 1 if number_of_device_registered >= 4 else 0
        is_high_spender = 1 if order_amount_hike > 20 else 0

        payload = {
            "Tenure": tenure,
            "CityTier": city_tier,
            "WarehouseToHome": warehouse_to_home,
            "Gender": gender_val,
            "HourSpendOnApp": hour_spend_on_app,
            "NumberOfDeviceRegistered": number_of_device_registered,
            "SatisfactionScore": satisfaction_score,
            "MaritalStatus": marital_val,
            "NumberOfAddress": number_of_address,
            "Complain": complain,
            "OrderAmountHikeFromlastYear": order_amount_hike,
            "CouponUsed": coupon_used,
            "OrderCount": order_count,
            "DaySinceLastOrder": day_since_last_order,
            "CashbackAmount": cashback_amount,
            "tenure_bucket": tenure_bucket,
            "engagement_score": engagement_score,
            "cashback_per_order": cashback_per_order,
            "is_recent_buyer": is_recent_buyer,
            "has_multi_device": has_multi_device,
            "is_high_spender": is_high_spender,
            "PreferredLoginDevice_Mobile Phone": 0,
            "PreferredPaymentMode_Credit Card": 0,
            "PreferredPaymentMode_Debit Card": 0,
            "PreferredPaymentMode_E wallet": 0,
            "PreferredPaymentMode_UPI": 0,
            "PreferedOrderCat_Grocery": 0,
            "PreferedOrderCat_Laptop & Accessory": 0,
            "PreferedOrderCat_Mobile Phone": 0,
            "PreferedOrderCat_Others": 0,
        }

        try:
            resp = requests.post(f"{FASTAPI_URL}/predict", json=payload, timeout=10)
            resp.raise_for_status()
            result = resp.json()
            return (
                f"Churn Prediction: {'Will Churn' if result['churn_prediction'] == 1 else 'Will Stay'}\n"
                f"Churn Probability: {result['churn_probability']:.1%}\n"
                f"Risk Level: {result['risk_level']}"
            )
        except requests.exceptions.ConnectionError:
            return (
                "Error: Cannot connect to the prediction API. "
                "Make sure FastAPI is running on " + FASTAPI_URL
            )
        except Exception as e:
            return f"Error calling prediction API: {e}"

    # -----------------------------------------------------------------------
    # Agent setup
    # -----------------------------------------------------------------------
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        max_output_tokens=2048,
    )

    tools = [query_dataframe, predict_churn]

    system_prompt = (
        "You are an AI churn analyst for an e-commerce platform. You help business "
        "users understand customer churn patterns and make predictions.\n\n"
        "You have access to two tools:\n"
        "1. query_dataframe — runs pandas code on the customer dataset (5,630 rows) "
        "to answer analytical questions. The DataFrame is called `df`.\n"
        "2. predict_churn — predicts churn probability for a specific customer "
        "profile by calling the ML model API.\n\n"
        "Guidelines:\n"
        "- For data questions, write clean pandas code. Use .to_string() for "
        "DataFrames to format output nicely.\n"
        "- When asked about churn rates by category, use "
        'df.groupby("column")["Churn"].mean()\n'
        "- For predictions, gather reasonable defaults for any features the user "
        "does not specify.\n"
        "- Always explain your findings in clear business language.\n"
        "- When giving recommendations, base them on data patterns you can query.\n"
        "- Format your responses with clear sections and bullet points where "
        "appropriate.\n"
        "- If a query fails, explain what went wrong and try a different approach."
    )

    agent_executor = create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
    )


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
layout = dbc.Container([
    # Header
    dbc.Row(dbc.Col(dbc.Card(dbc.CardBody([
        html.H5("AI Churn Analyst", className="mb-2"),
        html.P(
            "Ask questions about churn data or predict churn for specific "
            "customers. Powered by Google Gemini + LangChain.",
            className="text-muted mb-2",
        ),
        html.Div([
            dbc.Button(
                "Which payment method has highest churn?",
                id="ai-example-1", color="outline-primary", size="sm",
                className="me-2 mb-1",
            ),
            dbc.Button(
                "Predict churn for a customer with tenure 1 month",
                id="ai-example-2", color="outline-primary", size="sm",
                className="me-2 mb-1",
            ),
            dbc.Button(
                "Compare married vs single churn rates",
                id="ai-example-3", color="outline-primary", size="sm",
                className="me-2 mb-1",
            ),
            dbc.Button(
                "Top 3 actions to reduce churn?",
                id="ai-example-4", color="outline-primary", size="sm",
                className="mb-1",
            ),
        ]),
    ]), className="shadow-sm"), md=12), className="mb-3"),

    # Chat display
    dbc.Row(dbc.Col(dbc.Card(dbc.CardBody([
        dbc.Spinner(
            html.Div(
                id="ai-chat-display",
                style={"minHeight": "400px", "maxHeight": "500px", "overflowY": "auto"},
            ),
            color="primary",
            type="border",
            spinner_style={"width": "2rem", "height": "2rem"},
        ),
    ]), className="shadow-sm"), md=12), className="mb-3"),

    # Input area
    dbc.Row([
        dbc.Col(
            dbc.Input(
                id="ai-input", type="text",
                placeholder="Ask about churn patterns, predictions, or strategies...",
                debounce=False,
            ),
            md=10,
        ),
        dbc.Col(
            dbc.Button("Send", id="ai-send-btn", color="danger", className="w-100"),
            md=2,
        ),
    ], className="mb-3"),

    # Hidden stores
    dcc.Store(id="ai-chat-history", data=[]),
], fluid=True)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
def register_callbacks(app):
    # Callback 1: Send message
    @app.callback(
        Output("ai-chat-history", "data"),
        Output("ai-input", "value"),
        Input("ai-send-btn", "n_clicks"),
        Input("ai-input", "n_submit"),
        State("ai-input", "value"),
        State("ai-chat-history", "data"),
        prevent_initial_call=True,
    )
    def send_message(n_clicks, n_submit, user_input, chat_history):
        if not user_input or not user_input.strip():
            return no_update, no_update

        chat_history = chat_history or []
        chat_history.append({"role": "user", "content": user_input})

        if agent_executor is None:
            chat_history.append({
                "role": "assistant",
                "content": (
                    "The AI Analyst is not configured. Please set "
                    "`GOOGLE_API_KEY` in the `.env` file and restart the "
                    "dashboard.\n\nGet a free key at: "
                    "https://aistudio.google.com/apikey"
                ),
            })
            return chat_history, ""

        # Convert recent history to LangChain message format
        from langchain_core.messages import HumanMessage, AIMessage

        lc_messages = []
        for msg in chat_history[-11:-1]:
            if msg["role"] == "user":
                lc_messages.append(HumanMessage(content=msg["content"]))
            else:
                lc_messages.append(AIMessage(content=msg["content"]))
        lc_messages.append(HumanMessage(content=user_input))

        try:
            result = agent_executor.invoke({"messages": lc_messages})
            # Extract the last AI message from the response
            ai_messages = [
                m for m in result["messages"]
                if isinstance(m, AIMessage) and m.content
            ]
            assistant_response = ai_messages[-1].content if ai_messages else "No response generated."
        except Exception as e:
            assistant_response = f"I encountered an error processing your request: {e}"

        chat_history.append({"role": "assistant", "content": assistant_response})
        return chat_history, ""

    # Callback 2: Render chat messages
    @app.callback(
        Output("ai-chat-display", "children"),
        Input("ai-chat-history", "data"),
    )
    def render_chat(chat_history):
        if not chat_history:
            return html.Div([
                html.P(
                    "Welcome! Ask me anything about customer churn.",
                    className="text-muted text-center mt-5",
                ),
                html.P(
                    "Try clicking one of the example queries above.",
                    className="text-muted text-center",
                ),
            ])

        messages = []
        for msg in chat_history:
            if msg["role"] == "user":
                messages.append(
                    html.Div(
                        dbc.Card(
                            dbc.CardBody(html.P(msg["content"], className="mb-0")),
                            color="primary", inverse=True, className="ms-auto",
                            style={"maxWidth": "75%"},
                        ),
                        className="d-flex justify-content-end mb-2",
                    )
                )
            else:
                messages.append(
                    html.Div(
                        dbc.Card(
                            dbc.CardBody(
                                dcc.Markdown(msg["content"], className="mb-0")
                            ),
                            className="me-auto",
                            style={
                                "maxWidth": "85%",
                                "backgroundColor": "#f0f2f5",
                            },
                        ),
                        className="d-flex justify-content-start mb-2",
                    )
                )

        return html.Div(messages)

    # Callback 3: Example buttons populate input
    @app.callback(
        Output("ai-input", "value", allow_duplicate=True),
        Input("ai-example-1", "n_clicks"),
        Input("ai-example-2", "n_clicks"),
        Input("ai-example-3", "n_clicks"),
        Input("ai-example-4", "n_clicks"),
        prevent_initial_call=True,
    )
    def fill_example(n1, n2, n3, n4):
        trigger = callback_context.triggered_id
        examples = {
            "ai-example-1": "Which payment method has the highest churn rate?",
            "ai-example-2": (
                "Predict churn for a customer with tenure 1 month, "
                "satisfaction score 2, and who has complained"
            ),
            "ai-example-3": "Compare churn rates between married and single customers",
            "ai-example-4": (
                "What are the top 3 actions we should take to reduce churn "
                "based on the data?"
            ),
        }
        return examples.get(trigger, "")
