from flask import Flask, render_template, request, jsonify
import os, json, joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

MODEL_DIR = "models"
MODEL_NAMES = {
    "decision_tree": "Decision Tree",
    "knn": "K-Nearest Neighbors (KNN)",
    "logistic_regression": "Logistic Regression",
    "svm": "Support Vector Machine (SVM)"
}

def load_bundle(name):
    path = os.path.join(MODEL_DIR, f"{name}.joblib")
    if not os.path.exists(path):
        return None
    return joblib.load(path)

def make_row(form):
    values = {}
    values["Time"] = float(form.get("Time", 0))
    for i in range(1, 29):
        values[f"V{i}"] = float(form.get(f"V{i}", 0))
    values["Amount"] = float(form.get("Amount", 0))
    return pd.DataFrame([values])

def predict(model_key, row):
    bundle = load_bundle(model_key)
    if bundle is None:
        raise FileNotFoundError(
            "Models are not trained yet. Run: python train_models.py"
        )

    model = bundle["model"]
    feature_type = bundle["feature_type"]

    if feature_type == "original":
        X = row[bundle["features"]]
    elif feature_type == "knn_scaled":
        X0 = row[bundle["features"]]
        X = bundle["scaler"].transform(X0)
    elif feature_type == "svm":
        X = row.drop(columns=["Time"]).copy()
        X["std_Amount"] = bundle["amount_scaler"].transform(
            X["Amount"].values.reshape(-1, 1)
        )
        X = X.drop(columns=["Amount"])
        X = X[bundle["features"]]
    else:
        raise ValueError("Unknown model preprocessing.")

    pred = int(model.predict(X)[0])

    probability = None
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(X)[0][1])

    return pred, probability

@app.route("/")
def index():
    metrics = {}
    metrics_path = os.path.join(MODEL_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    return render_template("index.html", metrics=metrics, models=MODEL_NAMES)

@app.route("/detect")
def detect():
    return render_template("detect.html", models=MODEL_NAMES)

@app.route("/models")
def models():
    metrics = {}
    metrics_path = os.path.join(MODEL_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    return render_template("models.html", metrics=metrics)

@app.route("/about")
def about():
    return render_template("about.html")

@app.post("/predict")
def api_predict():
    try:
        model_key = request.form.get("model", "svm")
        if model_key not in MODEL_NAMES:
            return jsonify({"error": "Invalid model selected."}), 400

        row = make_row(request.form)
        pred, probability = predict(model_key, row)

        result = "Fraudulent" if pred == 1 else "Legitimate"
        risk = "HIGH" if pred == 1 else "LOW"

        if probability is not None:
            if probability >= 0.80:
                risk = "HIGH"
            elif probability >= 0.50:
                risk = "MEDIUM"
            else:
                risk = "LOW"

        return jsonify({
            "prediction": result,
            "class": pred,
            "probability": probability,
            "risk": risk,
            "model": MODEL_NAMES[model_key]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
