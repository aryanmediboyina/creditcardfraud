"""
Train the four models using preprocessing based on the supplied notebooks.

Put creditcard.csv.zip inside the data/ folder, then run:
    python train_models.py
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score
)

from imblearn.under_sampling import RandomUnderSampler

DATA_PATH = os.path.join("data", "creditcard.csv.zip")
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
print("Dataset loaded:", df.shape)

features_original = [c for c in df.columns if c != "Class"]

metrics = {}

# -------------------------
# 1. Decision Tree
# -------------------------
X = df.drop(columns="Class")
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30
)

dtree = DecisionTreeClassifier(
    criterion="entropy",
    random_state=0
)
dtree.fit(X_train, y_train)
pred = dtree.predict(X_test)

metrics["decision_tree"] = {
    "accuracy": accuracy_score(y_test, pred),
    "precision": precision_score(y_test, pred, zero_division=0),
    "recall": recall_score(y_test, pred, zero_division=0),
    "f1": f1_score(y_test, pred, zero_division=0)
}

joblib.dump({
    "model": dtree,
    "features": features_original,
    "feature_type": "original"
}, os.path.join(MODEL_DIR, "decision_tree.joblib"))

# -------------------------
# 2. KNN
# -------------------------
scaler_knn = StandardScaler()
X_knn = pd.DataFrame(
    scaler_knn.fit_transform(df.drop(columns=["Class"])),
    columns=features_original
)
y_knn = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X_knn, y_knn, test_size=0.30
)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)

metrics["knn"] = {
    "accuracy": accuracy_score(y_test, pred),
    "precision": precision_score(y_test, pred, zero_division=0),
    "recall": recall_score(y_test, pred, zero_division=0),
    "f1": f1_score(y_test, pred, zero_division=0)
}

joblib.dump({
    "model": knn,
    "scaler": scaler_knn,
    "features": features_original,
    "feature_type": "knn_scaled"
}, os.path.join(MODEL_DIR, "knn.joblib"))

# -------------------------
# 3. Logistic Regression
# -------------------------
legit = df[df.Class == 0]
fraud = df[df.Class == 1]

legit_sample = legit.sample(n=492)
new_dataset = pd.concat([legit_sample, fraud], axis=0)

X = new_dataset.drop(columns="Class")
y = new_dataset["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=2
)

logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
pred = logreg.predict(X_test)

metrics["logistic_regression"] = {
    "accuracy": accuracy_score(y_test, pred),
    "precision": precision_score(y_test, pred, zero_division=0),
    "recall": recall_score(y_test, pred, zero_division=0),
    "f1": f1_score(y_test, pred, zero_division=0)
}

joblib.dump({
    "model": logreg,
    "features": features_original,
    "feature_type": "original"
}, os.path.join(MODEL_DIR, "logistic_regression.joblib"))

# -------------------------
# 4. SVM
# -------------------------
svm_df = df.drop(columns=["Time"]).copy()

amount_scaler = StandardScaler()
svm_df["std_Amount"] = amount_scaler.fit_transform(
    svm_df["Amount"].values.reshape(-1, 1)
)
svm_df = svm_df.drop(columns=["Amount"])

cols = [c for c in svm_df.columns if c != "Class"]
X = svm_df[cols]
y = svm_df["Class"]

undersample = RandomUnderSampler(sampling_strategy=0.5)
X_under, y_under = undersample.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(
    X_under, y_under, test_size=0.2, random_state=1
)

svm = SVC(probability=True, random_state=2)
svm.fit(X_train, y_train)
pred = svm.predict(X_test)
proba = svm.predict_proba(X_test)[:, 1]

metrics["svm"] = {
    "accuracy": accuracy_score(y_test, pred),
    "precision": precision_score(y_test, pred, zero_division=0),
    "recall": recall_score(y_test, pred, zero_division=0),
    "f1": f1_score(y_test, pred, zero_division=0),
    "auc": roc_auc_score(y_test, proba)
}

joblib.dump({
    "model": svm,
    "amount_scaler": amount_scaler,
    "features": cols,
    "feature_type": "svm"
}, os.path.join(MODEL_DIR, "svm.joblib"))

with open(os.path.join(MODEL_DIR, "metrics.json"), "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

print("\nTraining complete!")
print("Models saved in:", MODEL_DIR)
print(json.dumps(metrics, indent=2))
