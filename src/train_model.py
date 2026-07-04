"""
train_model.py
---------------
Trains and evaluates classification models to predict customer churn.

Steps:
1. Load data and engineer features (encode categoricals, scale numerics)
2. Split into train/test sets
3. Train Logistic Regression (baseline) and Random Forest (main model)
4. Evaluate with accuracy, precision, recall, F1, ROC-AUC
5. Save the best-performing model + feature importance chart
"""

import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

DATA_PATH = "data/customer_churn.csv"
MODEL_PATH = "models/churn_model.joblib"
FIG_DIR = "reports/figures"

CATEGORICAL_COLS = [
    "gender",
    "partner",
    "dependents",
    "contract",
    "internet_service",
    "online_security",
    "tech_support",
    "paperless_billing",
    "payment_method",
]
NUMERIC_COLS = ["senior_citizen", "tenure_months", "monthly_charges", "total_charges"]


def load_and_prepare(path=DATA_PATH):
    df = pd.read_csv(path)
    y = (df["churn"] == "Yes").astype(int)
    X = df.drop(columns=["customer_id", "churn"])
    X = pd.get_dummies(X, columns=CATEGORICAL_COLS, drop_first=True)
    return X, y


def evaluate_model(name, model, X_test, y_test):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, probs),
    }
    print(f"\n--- {name} ---")
    for k, v in metrics.items():
        if k != "model":
            print(f"{k:>10}: {v:.3f}")
    print("\nClassification report:")
    print(classification_report(y_test, preds, target_names=["No Churn", "Churn"]))
    return metrics, preds


def plot_confusion_matrix(y_test, preds, name, filename):
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Churn", "Churn"])
    ax.set_yticklabels(["No Churn", "Churn"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix – {name}")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/{filename}", dpi=150)
    plt.close()


def plot_feature_importance(model, feature_names, top_n=12):
    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=True).tail(top_n)
    fig, ax = plt.subplots(figsize=(7, 5))
    importances.plot(kind="barh", color="#4C72B0", ax=ax)
    ax.set_title("Top Feature Importances (Random Forest)")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/feature_importance.png", dpi=150)
    plt.close()


def main():
    X, y = load_and_prepare()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Baseline model
    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    log_reg.fit(X_train_scaled, y_train)
    lr_metrics, lr_preds = evaluate_model(
        "Logistic Regression", log_reg, X_test_scaled, y_test
    )
    plot_confusion_matrix(y_test, lr_preds, "Logistic Regression", "confusion_matrix_lr.png")

    # Main model
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=8, random_state=42, class_weight="balanced"
    )
    rf.fit(X_train, y_train)
    rf_metrics, rf_preds = evaluate_model("Random Forest", rf, X_test, y_test)
    plot_confusion_matrix(y_test, rf_preds, "Random Forest", "confusion_matrix_rf.png")
    plot_feature_importance(rf, X.columns)

    # Pick the better model by ROC-AUC and persist it
    best_name, best_model, best_metrics = (
        ("random_forest", rf, rf_metrics)
        if rf_metrics["roc_auc"] >= lr_metrics["roc_auc"]
        else ("logistic_regression", log_reg, lr_metrics)
    )
    joblib.dump(
        {
            "model": best_model,
            "model_type": best_name,
            "feature_names": list(X.columns),
            "scaler": scaler if best_name == "logistic_regression" else None,
        },
        MODEL_PATH,
    )

    print(f"\nBest model: {best_name} (ROC-AUC={best_metrics['roc_auc']:.3f})")
    print(f"Saved to {MODEL_PATH}")

    results = pd.DataFrame([lr_metrics, rf_metrics]).set_index("model")
    results.to_csv("reports/model_comparison.csv")
    print("\nModel comparison:\n", results.round(3))


if __name__ == "__main__":
    main()
