"""Pydantic models for request/response schemas."""

from pydantic import BaseModel, Field


class CustomerFeatures(BaseModel):
    """Input features for a single customer prediction."""

    Tenure: float = Field(..., description="Months with the platform")
    CityTier: int = Field(..., ge=1, le=3, description="City tier (1, 2, or 3)")
    WarehouseToHome: float = Field(..., description="Distance from warehouse to home (km)")
    Gender: int = Field(..., ge=0, le=1, description="0=Male, 1=Female")
    HourSpendOnApp: float = Field(..., description="Hours spent on mobile app")
    NumberOfDeviceRegistered: int = Field(..., description="Number of devices registered")
    SatisfactionScore: int = Field(..., ge=1, le=5, description="Satisfaction score (1-5)")
    MaritalStatus: int = Field(..., ge=0, le=2, description="0=Single, 1=Married, 2=Divorced")
    NumberOfAddress: int = Field(..., description="Number of addresses registered")
    Complain: int = Field(..., ge=0, le=1, description="0=No complaint, 1=Complained")
    OrderAmountHikeFromlastYear: float = Field(..., description="Order amount hike from last year (%)")
    CouponUsed: float = Field(..., description="Number of coupons used")
    OrderCount: float = Field(..., description="Number of orders placed")
    DaySinceLastOrder: float = Field(..., description="Days since last order")
    CashbackAmount: float = Field(..., description="Cashback amount received")
    tenure_bucket: int = Field(..., ge=0, le=3, description="0=new, 1=settling, 2=established, 3=loyal")
    engagement_score: float = Field(..., description="HourSpendOnApp * OrderCount")
    cashback_per_order: float = Field(..., description="CashbackAmount / OrderCount")
    is_recent_buyer: int = Field(..., ge=0, le=1, description="1 if DaySinceLastOrder <= 3")
    has_multi_device: int = Field(..., ge=0, le=1, description="1 if devices >= 4")
    is_high_spender: int = Field(..., ge=0, le=1, description="1 if OrderAmountHike > 20")
    PreferredLoginDevice_Mobile_Phone: int = Field(0, alias="PreferredLoginDevice_Mobile Phone")
    PreferredPaymentMode_Credit_Card: int = Field(0, alias="PreferredPaymentMode_Credit Card")
    PreferredPaymentMode_Debit_Card: int = Field(0, alias="PreferredPaymentMode_Debit Card")
    PreferredPaymentMode_E_wallet: int = Field(0, alias="PreferredPaymentMode_E wallet")
    PreferredPaymentMode_UPI: int = Field(0)
    PreferedOrderCat_Grocery: int = Field(0)
    PreferedOrderCat_Laptop_Accessory: int = Field(0, alias="PreferedOrderCat_Laptop & Accessory")
    PreferedOrderCat_Mobile_Phone: int = Field(0, alias="PreferedOrderCat_Mobile Phone")
    PreferedOrderCat_Others: int = Field(0)

    model_config = {"populate_by_name": True}


class PredictionResponse(BaseModel):
    """Response for /predict endpoint."""

    churn_prediction: int = Field(..., description="0=Stayed, 1=Churned")
    churn_probability: float = Field(..., description="Probability of churn (0-1)")
    risk_level: str = Field(..., description="Low, Medium, or High")


class ExplanationResponse(BaseModel):
    """Response for /explain endpoint."""

    churn_prediction: int
    churn_probability: float
    base_value: float = Field(..., description="SHAP base value (average prediction)")
    feature_impacts: dict[str, float] = Field(..., description="SHAP values per feature, sorted by impact")


class WhatIfRequest(BaseModel):
    """Request for /what-if endpoint."""

    original: CustomerFeatures = Field(..., description="Original customer features")
    modifications: dict[str, float] = Field(..., description="Feature changes to simulate")


class WhatIfResponse(BaseModel):
    """Response for /what-if endpoint."""

    original_probability: float
    modified_probability: float
    probability_change: float = Field(..., description="Positive = higher churn risk")
    modifications: dict[str, float]


class HighRiskCustomer(BaseModel):
    """A single high-risk customer entry."""

    customer_index: int
    churn_probability: float
    risk_level: str
    top_risk_factors: dict[str, float]


class HighRiskResponse(BaseModel):
    """Response for /alerts/high-risk endpoint."""

    threshold: float
    count: int
    customers: list[HighRiskCustomer]


class ModelInfoResponse(BaseModel):
    """Response for /model-info endpoint."""

    model_name: str
    model_type: str
    n_features: int
    feature_names: list[str]
    pipeline_steps: list[str]
