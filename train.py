"""Real data pipeline: baseline -> models -> calibration -> business value -> fairness audit -> model card"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import CAT, ECON, FEATURES, NUM, customer_value, expected_value, load_clean

df = load_clean()
train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df.y)
Xtr, ytr, Xte, yte = train[FEATURES], train.y.values, test[FEATURES], test.y.values


def make_pre():
    return ColumnTransformer([("num", StandardScaler(), NUM), ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])


def scores(p):
    return {"roc_auc": float(roc_auc_score(yte, p)), "pr_auc": float(average_precision_score(yte, p)),
            "brier": float(brier_score_loss(yte, p))}


# ---- 1. Baseline: shudhu Contract dekhe (ek line er business rule) ----
rate = train.groupby("Contract").y.mean()
p_rule = test.Contract.map(rate).values

# ---- 2. Models: LogReg (C tune, CV) vs Gradient Boosting vs Calibrated GB ----
cv = StratifiedKFold(5, shuffle=True, random_state=42)
search = GridSearchCV(Pipeline([("pre", make_pre()), ("clf", LogisticRegression(max_iter=3000))]),
                      {"clf__C": [0.03, 0.1, 0.3, 1, 3]}, scoring="average_precision", cv=cv).fit(Xtr, ytr)
lr = search.best_estimator_
hgb = Pipeline([("pre", make_pre()), ("clf", HistGradientBoostingClassifier(
    learning_rate=0.05, max_depth=3, max_iter=150, l2_regularization=1.0, random_state=0))]).fit(Xtr, ytr)
hgb_cal = CalibratedClassifierCV(Pipeline([("pre", make_pre()), ("clf", HistGradientBoostingClassifier(
    learning_rate=0.05, max_depth=3, max_iter=150, l2_regularization=1.0, random_state=0))]),
    method="isotonic", cv=5).fit(Xtr, ytr)
probs = {"contract_rule": p_rule, "logreg": lr.predict_proba(Xte)[:, 1],
         "grad_boosting": hgb.predict_proba(Xte)[:, 1], "grad_boosting_calibrated": hgb_cal.predict_proba(Xte)[:, 1]}
models = {k: scores(p) for k, p in probs.items()}
print(f"best LogReg C={search.best_params_['clf__C']} (CV PR-AUC {search.best_score_:.3f})\n")
print(f"{'model':26s} {'ROC-AUC':>8s} {'PR-AUC':>8s} {'Brier':>7s}")
for k, v in models.items():
    print(f"{k:26s} {v['roc_auc']:8.3f} {v['pr_auc']:8.3f} {v['brier']:7.3f}")
p = probs["logreg"]   # SERVE: logreg (accuracy gb-r kachakachi + exact explanation + calibrated)

# ---- 3. Calibration: "70%" bolle ki sotti 70% churn kore? ----
bins = pd.qcut(p, 5, duplicates="drop")
calib = pd.DataFrame({"p": p, "y": yte}).groupby(bins, observed=True).agg(predicted=("p", "mean"), actual=("y", "mean"))
print("\ncalibration (5 bin):\n", calib.round(3).to_string(index=False))

# ---- 4. Business value: retention offer kake dibo? (test set e ashol label diye) ----
monthly = test.MonthlyCharges.values
gain = yte * ECON["save_rate"] * customer_value(monthly) - ECON["offer_cost"]   # contact korle prottek customer er net labh
top20 = np.zeros(len(p), bool)
top20[np.argsort(-p)[:int(0.2 * len(p))]] = True
strategies = {"contact_nobody": np.zeros(len(p), bool), "contact_everyone": np.ones(len(p), bool),
              "rule_month_to_month": (test.Contract == "Month-to-month").values,
              "model_top20pct": top20, "model_expected_value_rule": expected_value(p, monthly) > 0}
net = {k: {"net_profit": float(gain[m].sum()), "contacted_share": float(m.mean())} for k, m in strategies.items()}
net["oracle_contact_only_churners"] = {"net_profit": float(gain[yte == 1].sum()), "contacted_share": float(yte.mean())}
print(f"\nretention campaign net profit on {len(test)} test customers ($):")
for k, v in sorted(net.items(), key=lambda kv: -kv[1]["net_profit"]):
    print(f"  {k:30s} {v['net_profit']:9.0f}   (contacted {v['contacted_share']:.0%})")

# ---- 5. Fairness audit (gender model e use hoy ni, tobu check kori) ----
fair = {}
for col in ["gender", "SeniorCitizen"]:
    for g, idx in test.reset_index(drop=True).groupby(col).groups.items():
        idx = np.array(idx)
        fair[f"{col}={g}"] = {"n": int(len(idx)), "churn_rate": float(yte[idx].mean()), "roc_auc": float(roc_auc_score(yte[idx], p[idx])),
                              "mean_predicted": float(p[idx].mean()), "flagged_share_top20": float(top20[idx].mean())}
print("\nfairness audit:")
for k, v in fair.items():
    print(f"  {k:16s} n={v['n']:4d} churn={v['churn_rate']:.2f} auc={v['roc_auc']:.3f} predicted={v['mean_predicted']:.2f} flagged={v['flagged_share_top20']:.2f}")

# ---- 6. Global drivers ----
imp = permutation_importance(lr, Xte, yte, scoring="roc_auc", n_repeats=10, random_state=0)
drivers = dict(sorted(zip(FEATURES, imp.importances_mean.round(4).tolist()), key=lambda kv: -kv[1])[:8])
print("\ntop drivers (permutation, ROC-AUC drop):", drivers)

joblib.dump({"pipeline": lr, "features": FEATURES}, "model/churn_model.joblib")
metrics = {"models": models, "calibration": calib.round(4).to_dict("records"), "campaign": net, "fairness": fair,
           "drivers": drivers, "econ_assumptions": ECON, "n_train": len(train), "n_test": len(test)}
json.dump(metrics, open("model/metrics.json", "w"), indent=2)

card = f"""# Model Card: Telco Churn (logistic regression)
**Data:** IBM Telco Customer Churn sample dataset (7,043 customers, churn rate 26.5%). Split 80/20 stratified.
**Intended use:** retention campaign targeting (rank customers by churn risk). **Not** for denying service or pricing individuals.
**Performance (test):** ROC-AUC {models['logreg']['roc_auc']:.3f}, PR-AUC {models['logreg']['pr_auc']:.3f}, Brier {models['logreg']['brier']:.3f}
(contract-only rule baseline: ROC-AUC {models['contract_rule']['roc_auc']:.3f}).
**Why logistic regression:** accuracy gradient boosting-er kachakachi, probability calibrated, per-customer explanation exact.
**Excluded input:** gender, customerID. gender audit kora hoy (see metrics.json -> fairness).
**Business assumptions (bodlao):** {ECON}
**Limitations:** single snapshot data, kono time dimension nai; offer er asol effect (save_rate) ekta assumption, A/B test diye mapte hobe;
data ek company/region er, onno jaygay revalidate korte hobe.
"""
open("MODEL_CARD.md", "w").write(card)
print("\nsaved model + metrics.json + MODEL_CARD.md")
