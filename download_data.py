"""Data file na thakle (ba update korte chaile) eta chalao. Source: IBM sample dataset (Telco Customer Churn)."""
import urllib.request

URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
urllib.request.urlretrieve(URL, "data/Telco-Customer-Churn.csv")
print("downloaded -> data/Telco-Customer-Churn.csv")
