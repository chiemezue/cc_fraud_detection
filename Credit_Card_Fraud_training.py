# Import Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import joblib
import warnings

warnings.filterwarnings("ignore")


# Load the dataset
df = pd.read_csv(
    "creditcard.csv"
)  # Assuming 'creditcard.csv' is the name of your dataset file

# Display first five rows of the dataset
df.head()


df["Class"].value_counts()


# Check number of rows and columns
df.shape


# Display all column names
df.columns


# Check data types of each column
df.dtypes


# Check for missing values in each column
df.isnull().sum()


# Check number of duplicate rows
df.duplicated().sum()


# Remove duplicate rows
df = df.drop_duplicates()


# Display statistical summary of numerical features
df.describe()


# Separate features and target variable
X = df.drop("Class", axis=1)
y = df["Class"]


# Confirm feature and target dimensions
X.shape, y.shape


# Count number of fraud and non fraud transactions
y.value_counts()


# Calculate percentage distribution of classes
y.value_counts(normalize=True) * 100


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

Scaler = StandardScaler()


X_train = Scaler.fit_transform(X_train)


# Handling class imbalance using the SMOTE Technique
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)


# Visualize the distribution of fraud vs non fraud transactions
y.value_counts().plot(kind="bar")
plt.title("Class Distribution (Fraud vs Non Fraud)")
plt.xlabel("Class")
plt.ylabel("Number of Transactions")
plt.xticks(rotation=0)
plt.show()


# Separate fraud and non-fraud transactions
fraud = df[df["Class"] == 1]
non_fraud = df[df["Class"] == 0]


# Boxplot to compare transaction
sns.boxplot(x="Class", y="Amount", data=df)
plt.yscale("log")
plt.show()


# Distribution of all transaction times
df["Time"].hist(bins=50)
plt.title("Transaction Time Distribution")
plt.show()


# Correlation of features with fraud label
df.corr()["Class"].sort_values(ascending=False)


# Histogram comparison of transaction amounts
plt.hist(non_fraud["Amount"], bins=50, alpha=0.7, label="Non-Fraud")
plt.hist(fraud["Amount"], bins=50, alpha=0.7, label="Fraud")

# Log scale for better visualization
plt.yscale("log")

# Add labels and legend
plt.title("Amount Distribution")
plt.xlabel("Transaction Amount")
plt.ylabel("Frequency")
plt.legend()

plt.show()


# Scatter plot to visualize anomalies
plt.scatter(df["Amount"], df["V14"], alpha=0.3)

# Adding colour to separate fraud and non-fraud
plt.scatter(fraud["Amount"], fraud["V14"], color="red", label="Fraud")

# Customize plot
plt.xlabel("Transaction Amount")
plt.ylabel("V14 Feature")
plt.title("Fraud Anomaly Visualization")
plt.legend()

plt.show()


# Plot correlation heatmap
sns.heatmap(df.corr(), cmap="coolwarm")
plt.show()


# Summary
print("Total:", len(df))
print("Fraud:", df["Class"].sum())
print("Fraud %:", round(df["Class"].mean() * 100, 3))


# Model Creation
# The four classification models created and compared to determine the best model for this project
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# Random Forest
model_rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model_rf.fit(X_train_resampled, y_train_resampled)


# XGB classifier
model_xgb = XGBClassifier(random_state=42)
model_xgb.fit(X_train_resampled, y_train_resampled)


# Hyperparameter Tuning
from sklearn.model_selection import GridSearchCV  # importing the required library


# random forest
rf = RandomForestClassifier()
rf_grid = {
    "n_estimators": [50, 100],
    "max_depth": [None, 10],
    "min_samples_split": [2, 5],
}


# XGBoost
xgb = XGBClassifier(use_label_encoder=False, eva_metric="logloss")
xgb_grid = {
    "n_estimators": [50, 100],
    "max_depth": [3, 5],
    "learning_rate": [0.1, 0.01],
}


# n_estimators=100 is the default and usually perfect
rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)


## Evaluation metrics for Random Forest
from sklearn.metrics import classification_report, confusion_matrix

print("--- RANDOM FOREST ---")
print(f"Precision: {precision_score(y_test, y_pred_rf):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_rf):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred_rf):.4f}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))


y_pred_rf
cm = confusion_matrix(y_test, y_pred_rf)
print(cm)

cm = confusion_matrix(y_test, y_pred_rf)

plt.imshow(cm)
plt.colorbar()
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center")


print("Random Forest")
print(classification_report(y_test, y_pred_rf))


# This model tells the model that fraud is rare
weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

xgb = XGBClassifier(n_estimators=100, scale_pos_weight=weight)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

y_pred_xgb

## Evaluation metrics for XGBClassifier
print("\n--- XGBOOST ---")
print(f"Precision: {precision_score(y_test, y_pred_xgb):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_xgb):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred_xgb):.4f}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_xgb))


cm = confusion_matrix(y_test, y_pred_xgb)

plt.imshow(cm)
plt.colorbar()
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center")


print("XGBoost")
print(classification_report(y_test, y_pred_xgb))


from sklearn.ensemble import IsolationForest

# contamination=0.01 tells it to flag the top 1% most unusual cases
iso = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
iso.fit(X_train)

# Convert -1 to 1 (fraud) and 1 to 0 (normal)
y_pred_iso = [1 if x == -1 else 0 for x in iso.predict(X_test)]

## Evaluation metrics for ISOLATION FORE
print("\n--- ISOLATION FOREST ---")
print(f"Precision: {precision_score(y_test, y_pred_iso):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_iso):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred_iso):.4f}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_iso))


y_pred_iso

cm = confusion_matrix(y_test, y_pred_iso)

plt.imshow(cm)
plt.colorbar()
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center")


print("Isolation Forest")
print(classification_report(y_test, y_pred_iso))


## Evaluating the best performing model
print("RF:", np.bincount(y_pred_rf))
print("XGB:", np.bincount(y_pred_xgb))
print("ISO:", np.bincount(y_pred_iso))


## Visualize Results
from sklearn.metrics import roc_curve, auc


models = {"Random Forest": rf, "XGBoost": xgb, "Isolation Forest": iso}
plt.figure(figsize=(8, 6))


models = {
    "Random Forest": y_pred_rf,
    "XGBoost": y_pred_xgb,
    "Isolation Forest": y_pred_iso,
}


# Confusion Matrix Heatmaps
for name, y_pred in models.items():
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {name}")
    plt.show()


## Saving the model for deployment or future use
joblib.dump(xgb, "xgb_fraud_model.pkl")  # replace xgb with your best model

# ✅ SAVE THE SCALER TOO (ADD THIS LINE)
joblib.dump(Scaler, "scaler.pkl")
