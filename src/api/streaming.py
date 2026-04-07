"""Streaming data simulator — generates synthetic customer events."""

import random
import time
from collections import deque
from datetime import datetime

from fastapi import APIRouter

router = APIRouter()

# In-memory event queue (last 100 events)
event_queue: deque = deque(maxlen=100)

# Realistic value ranges based on dataset statistics
FEATURE_RANGES = {
    "Tenure": (0, 61),
    "CityTier": [1, 2, 3],
    "WarehouseToHome": (5, 36),
    "Gender": [0, 1],
    "HourSpendOnApp": (0, 5),
    "NumberOfDeviceRegistered": (1, 6),
    "SatisfactionScore": [1, 2, 3, 4, 5],
    "MaritalStatus": [0, 1, 2],
    "NumberOfAddress": (1, 22),
    "Complain": [0, 1],
    "OrderAmountHikeFromlastYear": (11, 26),
    "CouponUsed": (0, 16),
    "OrderCount": (1, 16),
    "DaySinceLastOrder": (0, 46),
    "CashbackAmount": (0, 325),
}


def _generate_customer() -> dict:
    """Generate a random customer with realistic feature values."""
    customer = {}
    for feat, vals in FEATURE_RANGES.items():
        if isinstance(vals, list):
            customer[feat] = random.choice(vals)
        else:
            low, high = vals
            customer[feat] = round(random.uniform(low, high), 2)

    # Derived features
    tenure = customer["Tenure"]
    if tenure <= 6:
        customer["tenure_bucket"] = 0
    elif tenure <= 12:
        customer["tenure_bucket"] = 1
    elif tenure <= 24:
        customer["tenure_bucket"] = 2
    else:
        customer["tenure_bucket"] = 3

    customer["engagement_score"] = round(customer["HourSpendOnApp"] * customer["OrderCount"], 2)
    customer["cashback_per_order"] = (
        round(customer["CashbackAmount"] / customer["OrderCount"], 2)
        if customer["OrderCount"] > 0
        else 0
    )
    customer["is_recent_buyer"] = 1 if customer["DaySinceLastOrder"] <= 3 else 0
    customer["has_multi_device"] = 1 if customer["NumberOfDeviceRegistered"] >= 4 else 0
    customer["is_high_spender"] = 1 if customer["OrderAmountHikeFromlastYear"] > 20 else 0

    # One-hot encoded features (random selection)
    login_device = random.choice(["Computer", "Mobile Phone"])
    customer["PreferredLoginDevice_Mobile Phone"] = 1 if login_device == "Mobile Phone" else 0

    payment = random.choice(["Cash on Delivery", "Credit Card", "Debit Card", "E wallet", "UPI"])
    customer["PreferredPaymentMode_Credit Card"] = 1 if payment == "Credit Card" else 0
    customer["PreferredPaymentMode_Debit Card"] = 1 if payment == "Debit Card" else 0
    customer["PreferredPaymentMode_E wallet"] = 1 if payment == "E wallet" else 0
    customer["PreferredPaymentMode_UPI"] = 1 if payment == "UPI" else 0

    category = random.choice(["Fashion", "Grocery", "Laptop & Accessory", "Mobile Phone", "Others"])
    customer["PreferedOrderCat_Grocery"] = 1 if category == "Grocery" else 0
    customer["PreferedOrderCat_Laptop & Accessory"] = 1 if category == "Laptop & Accessory" else 0
    customer["PreferedOrderCat_Mobile Phone"] = 1 if category == "Mobile Phone" else 0
    customer["PreferedOrderCat_Others"] = 1 if category == "Others" else 0

    return customer


@router.post("/generate")
def generate_event():
    """Generate a single synthetic customer event and add to the queue."""
    customer = _generate_customer()
    event = {
        "timestamp": datetime.now().isoformat(),
        "customer": customer,
    }
    event_queue.append(event)
    return {"status": "generated", "event": event}


@router.post("/generate-batch")
def generate_batch(count: int = 10):
    """Generate multiple synthetic customer events."""
    events = []
    for _ in range(min(count, 50)):
        customer = _generate_customer()
        event = {
            "timestamp": datetime.now().isoformat(),
            "customer": customer,
        }
        event_queue.append(event)
        events.append(event)
    return {"status": "generated", "count": len(events), "events": events}


@router.get("/events")
def get_events(limit: int = 20):
    """Get recent events from the queue."""
    events = list(event_queue)[-limit:]
    return {"count": len(events), "events": events}


@router.get("/stats")
def get_stats():
    """Get streaming statistics."""
    return {
        "total_events": len(event_queue),
        "queue_capacity": event_queue.maxlen,
    }
