import json

from fastapi.testclient import TestClient

from app.main import app
from common import FEATURES, load_clean

client = TestClient(app)
M = json.load(open("model/metrics.json"))
HIGH = {"tenure": 1, "monthly_charges": 95, "partner": "No", "dependents": "No", "phone_service": "Yes",
        "multiple_lines": "No", "internet_service": "Fiber optic", "online_security": "No", "online_backup": "No",
        "device_protection": "No", "tech_support": "No", "streaming_tv": "Yes", "streaming_movies": "Yes",
        "contract": "Month-to-month", "paperless_billing": "Yes", "payment_method": "Electronic check"}
LOW = {"tenure": 60, "monthly_charges": 45, "partner": "Yes", "dependents": "Yes", "phone_service": "Yes",
       "multiple_lines": "No", "internet_service": "DSL", "online_security": "Yes", "online_backup": "Yes",
       "device_protection": "Yes", "tech_support": "Yes", "streaming_tv": "No", "streaming_movies": "No",
       "contract": "Two year", "paperless_billing": "No", "payment_method": "Credit card (automatic)"}


# ---- Data ----
def test_no_leakage_or_sensitive_features():
    assert not {"customerID", "Churn", "y", "gender"} & set(FEATURES)


def test_cleaning_fixes_blank_total_charges():
    df = load_clean()
    assert df.TotalCharges.isna().sum() == 0 and not df.customerID.duplicated().any() and len(df) == 7043


# ---- Model quality gates (CI e fail korle deploy hobe na) ----
def test_quality_gate_and_beats_business_rule():
    m = M["models"]
    assert m["logreg"]["roc_auc"] >= 0.82
    assert m["logreg"]["roc_auc"] > m["contract_rule"]["roc_auc"] + 0.08
    assert abs(m["logreg"]["roc_auc"] - m["grad_boosting"]["roc_auc"]) < 0.015   # simple model e-i jothesto


def test_calibration_within_tolerance():
    assert all(abs(b["predicted"] - b["actual"]) < 0.06 for b in M["calibration"])


def test_campaign_value_ordering():
    c = M["campaign"]
    ev, rule, everyone = c["model_expected_value_rule"], c["rule_month_to_month"], c["contact_everyone"]
    assert ev["net_profit"] > rule["net_profit"] > everyone["net_profit"] > 0 == c["contact_nobody"]["net_profit"]


def test_fairness_gaps_bounded():
    f = M["fairness"]
    assert abs(f["gender=Female"]["roc_auc"] - f["gender=Male"]["roc_auc"]) < 0.03
    assert min(f["SeniorCitizen=0"]["roc_auc"], f["SeniorCitizen=1"]["roc_auc"]) > 0.75


# ---- API ----
def test_high_vs_low_risk_and_reasons():
    h, l = client.post("/predict", json=HIGH).json(), client.post("/predict", json=LOW).json()
    assert h["churn_probability"] > 0.5 > 0.2 > l["churn_probability"]
    assert h["risk_tier"] == "high" and l["risk_tier"] == "low"
    assert h["recommended_action"] == "offer_retention" and l["recommended_action"] == "no_action"
    assert "Contract" in [r["feature"] for r in h["reasons"]["increases_risk"]]
    assert "Contract" in [r["feature"] for r in l["reasons"]["reduces_risk"]]


def test_batch_validation_and_homepage():
    assert client.post("/predict-batch", json=[HIGH, LOW]).json()["count"] == 2
    assert client.post("/predict", json={**HIGH, "contract": "Weekly"}).status_code == 422
    r = client.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
