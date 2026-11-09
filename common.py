"""Shared: data cleaning, feature list, business economics, per-customer explanation"""
import numpy as np
import pandas as pd

NUM = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
CAT = ["Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
       "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
       "PaperlessBilling", "PaymentMethod"]
FEATURES = NUM + CAT          # NOTE: customerID, gender, Churn model e jay na (gender: audit er jonno rakha, input hishebe na)

# Business assumption (nijer company onujayi bodlao): retention offer er economics
ECON = {"save_rate": 0.30,    # offer dile 30% churner thake
        "months": 12,         # thake gele aro koto mash revenue
        "margin": 0.35,       # revenue er koto % labh
        "offer_cost": 15.0}   # prottek contact/offer e khoroch ($)


def load_clean(path="data/Telco-Customer-Churn.csv"):
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")   # 11ta blank string " " ache real data te
    df.loc[df.tenure == 0, "TotalCharges"] = df.loc[df.tenure == 0, "TotalCharges"].fillna(0.0)  # notun customer, ekhono bill nai
    df["y"] = (df["Churn"] == "Yes").astype(int)
    return df


def customer_value(monthly):
    return np.asarray(monthly) * ECON["months"] * ECON["margin"]


def expected_value(p, monthly):
    """Contact korle expected labh = p(churn) x save_rate x customer value - offer cost"""
    return np.asarray(p) * ECON["save_rate"] * customer_value(monthly) - ECON["offer_cost"]


def explain(pipe, row, top=3):
    """Logistic regression er exact per-customer contribution (log-odds). + = churn risk bare, - = kome"""
    pre, clf = pipe.named_steps["pre"], pipe.named_steps["clf"]
    xt = pre.transform(row)
    xt = xt.toarray() if hasattr(xt, "toarray") else np.asarray(xt)
    contrib = xt[0] * clf.coef_[0]
    groups = {}
    for name, c in zip(pre.get_feature_names_out(), contrib):
        raw = name.split("__", 1)[1]
        col = next((k for k in CAT if raw.startswith(k + "_")), raw)
        groups[col] = groups.get(col, 0.0) + float(c)
    items = [{"feature": k, "value": str(row.iloc[0][k]), "impact": round(v, 3)} for k, v in groups.items()]
    items.sort(key=lambda d: -d["impact"])
    return {"increases_risk": [d for d in items[:top] if d["impact"] > 0],
            "reduces_risk": [d for d in items[::-1][:top] if d["impact"] < 0]}
