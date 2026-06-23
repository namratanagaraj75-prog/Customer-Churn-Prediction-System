import pandas as pd

df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("Total Customers:", len(df))

print("\nChurn Count:")
print(df["Churn"].value_counts())

print("\nChurn Percentage:")
print(round(df["Churn"].value_counts(normalize=True) * 100, 2))