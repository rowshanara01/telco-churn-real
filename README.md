# Telco Customer Churn (REAL data, end-to-end)

Data: IBM Telco Customer Churn sample dataset, 7,043 ashol customer (data/Telco-Customer-Churn.csv).
Ager synthetic project gulor sathe pharak: real data messy, signal durbol, "bhalo AUC" 0.84, 1.0 na.

## Chalanor niyom
pip install -r requirements.txt
python eda.py            # data bujhi: quality check, churn kothay beshi
python train.py          # baseline vs models vs calibration vs business value vs fairness + MODEL_CARD.md
pytest -q                # data test + model quality gate + API test
uvicorn app.main:app --reload     # browser e http://localhost:8000 (UI), /docs (API)
(data na thakle: python download_data.py)

## Real-world engineer ja ja kore (ei project e ache)
1. Data quality: TotalCharges e 11ta blank string (shob tenure=0), duplicate check, leakage (customerID/gender bad)
2. Baseline age: "shudhu Contract dekho" rule (AUC 0.74). Model eta er cheye koto bhalo?
3. Simple vs complex: LogReg (0.842) ~ Gradient Boosting (0.844). Simple model serve kora, karon explainable
4. Calibration: model "60%" bolle sotti-i ~60% churn kore kina (table dekho)
5. Business value: accuracy na, TAKA. Retention campaign e kake offer dile net profit beshi?
   expected_value = p(churn) x save_rate x customer_value - offer_cost
6. Per-customer explanation: keno high risk (exact logistic contribution)
7. Fairness audit: gender, SeniorCitizen group wise AUC/flag rate
8. Model card: intended use, limitation, assumption
9. Quality gate test: AUC kom hole pytest fail -> deploy hobe na
10. Web UI + API + Docker

## Shimabodhdhota (sotti kotha)
- save_rate (offer e koto % thake) ekta ASSUMPTION. Asol-e A/B test kore mapte hobe.
- SeniorCitizen group e AUC kom (0.78 vs 0.84) ar beshi flagged (churn rate beshi bole). Real project e eta niye team er sathe alochona kora dorkar.
- Data single snapshot, time dimension nai.

## Challenge
1. Survival analysis: "kobe" chhere jabe (time-to-churn), shudhu "chhere jabe kina" na
2. Uplift modeling (uplift-coupon project) real A/B data diye
3. SHAP add koro, HGB er explanation
4. ECON assumption bodlao, kon strategy jite dekho
5. Next.js frontend banao (static/index.html er jaygay)
6. Drift monitoring (fraud-mlops er monitor.py reuse)
