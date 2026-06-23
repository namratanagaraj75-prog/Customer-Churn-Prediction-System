import pandas as pd

df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

print(
    df.groupby("Churn")["tenure"]
      .mean()
      .round(2)
)