"""Churn API: probability + risk tier + KENO (reasons) + retention offer dewa laabhjonok kina"""
import json
from typing import Literal, Optional

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from common import ECON, FEATURES, customer_value, expected_value, explain

B = joblib.load("model/churn_model.joblib")
PIPE = B["pipeline"]
app = FastAPI(title="Telco Churn API (real data)")
YN = Literal["Yes", "No"]
ADDON = Literal["Yes", "No", "No internet service"]


class Customer(BaseModel):
    tenure: int = Field(ge=0, le=100, description="koto mash dhore customer")
    monthly_charges: float = Field(gt=0, le=500)
    total_charges: Optional[float] = Field(default=None, ge=0, description="na dile tenure x monthly")
    senior_citizen: Literal[0, 1] = 0
    partner: YN
    dependents: YN
    phone_service: YN
    multiple_lines: Literal["Yes", "No", "No phone service"]
    internet_service: Literal["DSL", "Fiber optic", "No"]
    online_security: ADDON
    online_backup: ADDON
    device_protection: ADDON
    tech_support: ADDON
    streaming_tv: ADDON
    streaming_movies: ADDON
    contract: Literal["Month-to-month", "One year", "Two year"]
    paperless_billing: YN
    payment_method: Literal["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]


MAP = {"tenure": "tenure", "monthly_charges": "MonthlyCharges", "total_charges": "TotalCharges",
       "senior_citizen": "SeniorCitizen", "partner": "Partner", "dependents": "Dependents",
       "phone_service": "PhoneService", "multiple_lines": "MultipleLines", "internet_service": "InternetService",
       "online_security": "OnlineSecurity", "online_backup": "OnlineBackup", "device_protection": "DeviceProtection",
       "tech_support": "TechSupport", "streaming_tv": "StreamingTV", "streaming_movies": "StreamingMovies",
       "contract": "Contract", "paperless_billing": "PaperlessBilling", "payment_method": "PaymentMethod"}


def to_frame(c: Customer):
    d = {MAP[k]: v for k, v in c.model_dump().items()}
    if d["TotalCharges"] is None:
        d["TotalCharges"] = d["tenure"] * d["MonthlyCharges"]
    return pd.DataFrame([d])[FEATURES]


def tier(p):
    return "high" if p >= 0.5 else "medium" if p >= 0.2 else "low"


def score(c: Customer):
    row = to_frame(c)
    p = float(PIPE.predict_proba(row)[0, 1])
    ev = float(expected_value(p, c.monthly_charges))
    return {"churn_probability": round(p, 4), "risk_tier": tier(p),
            "customer_value_12m": round(float(customer_value(c.monthly_charges)), 2),
            "expected_value_if_offer": round(ev, 2),
            "recommended_action": "offer_retention" if ev > 0 else "no_action",
            "reasons": explain(PIPE, row)}


@app.get("/", include_in_schema=False)
def home():
    return FileResponse("static/index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    return json.load(open("model/metrics.json"))


@app.post("/predict")
def predict(c: Customer):
    return score(c)


@app.post("/predict-batch")
def predict_batch(cs: list[Customer]):
    return {"count": len(cs), "results": [score(c) for c in cs], "assumptions": ECON}
