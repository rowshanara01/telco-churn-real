"""Model banano-r age data bujhi: quality check + churn kothay beshi"""
import pandas as pd

from common import load_clean

raw = pd.read_csv("data/Telco-Customer-Churn.csv")
df = load_clean()
print("rows:", len(df), "| duplicate customerID:", df.customerID.duplicated().sum())
print("blank TotalCharges in raw:", int((pd.to_numeric(raw.TotalCharges, errors='coerce').isna()).sum()),
      "(sob tenure=0 customer)")
print(f"overall churn rate: {df.y.mean():.1%}  <- imbalanced\n")
for col in ["Contract", "InternetService", "PaymentMethod", "TechSupport"]:
    print(df.groupby(col).y.agg(["mean", "size"]).round(3).rename(columns={"mean": "churn_rate", "size": "customers"}), "\n")
df["tenure_bucket"] = pd.cut(df.tenure, [-1, 6, 12, 24, 48, 100], labels=["0-6", "7-12", "13-24", "25-48", "49+"])
print(df.groupby("tenure_bucket", observed=True).y.mean().round(3).rename("churn_rate"))
