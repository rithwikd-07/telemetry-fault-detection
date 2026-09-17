import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


# 1. Load dataset
df = pd.read_csv("data/predictive_maintenance.csv")

print("=== DATASET INFORMATION ===")
print("Dataset shape:", df.shape)

print("\nMissing values:")
print(df.isnull().sum())

print("\nMachine failure distribution:")
print(df["Machine failure"].value_counts())


# 2. Select telemetry features
features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

target = "Machine failure"

print("\n=== DATA PREPARATION ===")
print("Features used:")
print(features)

print("\nTarget:")
print(target)


# 3. Separate features and target
X = df[features]
y = df[target]


# 4. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())


# 5. Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)


# 6. Train model
print("\n=== MODEL TRAINING ===")
print("Training Random Forest model...")

model.fit(X_train, y_train)


# 7. Make predictions
y_pred = model.predict(X_test)


# 8. Evaluate model
print("\n=== MODEL PERFORMANCE ===")

print(classification_report(
    y_test,
    y_pred,
    target_names=["Normal", "Failure"],
    zero_division=0
))


# 9. Create output folders
os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)


# 10. Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=["Normal", "Failure"],
    yticklabels=["Normal", "Failure"]
)

plt.title("Machine Failure Prediction - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(
    "outputs/confusion_matrix.png",
    dpi=300
)

plt.close()

print("Confusion matrix saved to: outputs/confusion_matrix.png")


# 11. Create feature importance graph
importance = pd.Series(
    model.feature_importances_,
    index=features
)

importance = importance.sort_values(ascending=True)

plt.figure(figsize=(8, 5))

importance.plot(kind="barh")

plt.title("Feature Importance - Machine Failure Prediction")
plt.xlabel("Importance")

plt.tight_layout()

plt.savefig(
    "outputs/feature_importance.png",
    dpi=300
)

plt.close()

print("Feature importance saved to: outputs/feature_importance.png")


# 12. Save trained model
joblib.dump(
    model,
    "models/random_forest.pkl"
)

print("Model saved to: models/random_forest.pkl")

print("\n=== PROJECT TRAINING COMPLETED ===")