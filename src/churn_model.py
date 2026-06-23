import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load cleaned data
df = pd.read_csv("data/cleaned_telco_churn.csv")

# Drop customer ID
df = df.drop("customerID", axis=1)

# Encode all categorical columns
label_encoder = LabelEncoder()

for col in df.select_dtypes(include=["object"]).columns:
    df[col] = label_encoder.fit_transform(df[col])

# Features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model
model = LogisticRegression(max_iter=5000)

# Train
print(X.dtypes)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Results
print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))