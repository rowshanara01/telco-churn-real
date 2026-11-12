## Telco Customer Churn (REAL Data, End-to-End) ##

**Dataset:** IBM Telco Customer Churn Sample Dataset  
**Records:** 7,043 real customer records  
**File:** `data/Telco-Customer-Churn.csv`

Unlike previous synthetic projects, this project uses real-world data, which is often messy and contains weak or imperfect signals. A good AUC score is around **0.84**, not 1.0.

## How to Run the Project ##

Install the dependencies:

pip install -r requirements.txt

Run Exploratory Data Analysis (EDA):

python eda.py

Understand the data, check data quality, and identify where customer churn is highest.

Train the models:

python train.py

Compare baseline and machine learning models, evaluate calibration, calculate business value, perform fairness analysis, and generate MODEL_CARD.md.

Run the tests:

pytest -q

Start the API and web application:

uvicorn app.main:app --reload

Open the following URLs in your browser:

- Web UI: http://localhost:8000
- API Documentation: http://localhost:8000/docs

If the dataset is missing, download it first:

python download_data.py

## What Real-World Engineers Do in This Project ##

### 1. Data Quality Checks

- Identify 11 blank strings in TotalCharges (all belong to customers with tenure = 0).
- Check for duplicate records.
- Prevent data leakage by excluding customerID and gender from model training.

### 2. Establish a Baseline Before Training ###

Start with a simple rule:

"Only look at the Contract column."

This baseline achieves an AUC of approximately 0.74.

The goal is to determine how much better the machine learning model performs compared to this simple rule.

### 3. Compare Simple and Complex Models

Compare different machine learning algorithms:

| Model | AUC |
|---|---:|
| Logistic Regression | 0.842 |
| Gradient Boosting | 0.844 |

The performance difference is very small.

Therefore, choose Logistic Regression for production because it is simpler, easier to explain, and more interpretable.

### 4. Model Calibration

Check whether the predicted probabilities reflect actual customer behavior.

For example:

If the model predicts a 60% churn probability for a group of customers, approximately 60% of those customers should actually churn.

Check the calibration table to evaluate this.

### 5. Business Value: Focus on Money, Not Just Accuracy ##

The objective is not simply to achieve high accuracy. The objective is to generate business value.

Determine which customers should receive retention offers to maximize net profit.

Expected Value = P(Churn) × Save Rate × Customer Value - Offer Cost

This helps identify customers who are worth targeting with retention campaigns.

### 6. Per-Customer Explanations ##3

Explain why an individual customer has a high churn risk.

Use the exact Logistic Regression feature contributions to identify which factors increase or decrease that customer's predicted risk.

### 7. Fairness Audit

Evaluate model performance across different demographic groups.

Analyze:

- Gender-wise AUC.
- SeniorCitizen-wise AUC.
- Prediction or flag rates across groups.

Identify whether the model performs differently for different customer groups.

### 8. Model Card ###

Generate a MODEL_CARD.md file documenting:

- Intended use.
- Model limitations.
- Assumptions.
- Performance.
- Potential risks.

### 9. Automated Quality Gate

Use automated tests to ensure the model meets minimum quality requirements.

If the AUC falls below the required threshold, pytest fails.

This prevents a model that does not meet quality standards from being deployed.

### 10. Web UI, API, and Docker

The project includes:

- Web-based user interface.
- FastAPI backend.
- API documentation.
- Docker support.

This makes the project closer to a real-world machine learning application.

## Project Limitations (Being Honest About the Data)

1. Save Rate Is an Assumption:
The percentage of customers who stay after receiving a retention offer is assumed. It must be measured through real A/B testing.

2. Fairness Concern:
The SeniorCitizen group has lower AUC (0.78 vs. 0.84) and a higher prediction flag rate because its churn rate is higher. In a real-world project, this requires discussion with the team and further investigation.

3. Single Data Snapshot:
The dataset represents a single snapshot and does not contain a time dimension. Therefore, it cannot directly predict when a customer will leave.

## Advanced Challenges

### 1. Survival Analysis

Predict when a customer is likely to leave, rather than simply predicting whether they will leave.

Explore time-to-event modeling.

### 2. Uplift Modeling

Implement uplift modeling using real A/B testing data from the uplift-coupon project.

The goal is to identify which customers are more likely to stay because of a retention offer.

### 3. SHAP Explanations

Add SHAP to explain the predictions of the HistGradientBoosting (HGB) model.

### 4. Economic Assumption Analysis

Change the economic assumptions, such as:

- Retention save rate.
- Customer lifetime value.
- Offer cost.

Compare different retention strategies to determine which one generates the highest profit.

### 5. Next.js Frontend

Replace the existing static/index.html interface with a modern Next.js frontend.

### 6. Drift Monitoring

Implement data and model drift monitoring.

Reuse the monitor.py module from the fraud-mlops project.

---

## Project Goal

Build an end-to-end, production-oriented machine learning system that demonstrates real-world data analysis, model development, explainability, fairness, business decision-making, automated testing, API development, and deployment.
