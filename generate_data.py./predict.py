"""
predict.py
----------
Loads the trained model and predicts churn probability for new customers.

Usage:
    python src/predict.py
"""

import joblib
import pandas as pd

MODEL_PATH = "models/churn_model.joblib"


def build_feature_row(sample: dict, feature_names: list) -> pd.DataFrame:
    """Convert a raw customer dict into the one-hot-encoded feature row
    the model expects, filling any missing dummy columns with 0."""
    df = pd.DataFrame([sample])
    df = pd.get_dummies(df)
    df = df.reindex(columns=feature_names, fill_value=0)
    return df


def predict_churn(sample: dict) -> dict:
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    feature_names = bundle["feature_names"]
    scaler = bundle["scaler"]

    X = build_feature_row(sample, feature_names)
    if scaler is not None:
        X = scaler.transform(X)

    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]
    return {"churn_prediction": "Yes" if pred == 1 else "No", "churn_probability": round(float(prob), 3)}


if __name__ == "__main__":
    # Example: a newer customer on a month-to-month contract, no tech support
    sample_customer = {
        "gender": "Female",
        "senior_citizen": 0,
        "partner": "No",
        "dependents": "No",
        "tenure_months": 3,
        "contract": "Month-to-month",
        "internet_service": "Fiber optic",
        "online_security": "No",
        "tech_support": "No",
        "paperless_billing": "Yes",
        "payment_method": "Electronic check",
        "monthly_charges": 89.5,
        "total_charges": 268.5,
    }

    result = predict_churn(sample_customer)
    print("Sample customer:")
    for k, v in sample_customer.items():
        print(f"  {k}: {v}")
    print("\nPrediction:")
    print(f"  Will churn?      {result['churn_prediction']}")
    print(f"  Churn probability: {result['churn_probability']}")
