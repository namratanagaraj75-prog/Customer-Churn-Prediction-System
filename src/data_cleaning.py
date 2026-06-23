import pandas as pd

df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Check data types
print(df.dtypes)

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# TotalCharges often contains blank strings
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

print("\nMissing Values After Conversion:")
print(df["TotalCharges"].isnull().sum())

# Remove rows with missing TotalCharges
df = df.dropna()

print("\nFinal Shape:")
print(df.shape)

# Save cleaned dataset
df.to_csv(
    "data/cleaned_telco_churn.csv",
    index=False
)

print("\nCleaned dataset saved successfully.")