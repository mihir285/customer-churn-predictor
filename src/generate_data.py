import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_CUSTOMERS = 3000


def generate_customers(n=N_CUSTOMERS, seed=RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    customer_id = [f"CUST-{i:05d}" for i in range(1, n + 1)]
    gender = rng.choice(["Male", "Female"], size=n)
    senior_citizen = rng.choice([0, 1], size=n, p=[0.84, 0.16])
    partner = rng.choice(["Yes", "No"], size=n, p=[0.48, 0.52])
    dependents = rng.choice(["Yes", "No"], size=n, p=[0.30, 0.70])

    tenure_months = rng.integers(0, 73, size=n)

    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"], size=n, p=[0.55, 0.24, 0.21]
    )
    internet_service = rng.choice(
        ["DSL", "Fiber optic", "No"], size=n, p=[0.34, 0.44, 0.22]
    )
    online_security = rng.choice(["Yes", "No"], size=n, p=[0.29, 0.71])
    tech_support = rng.choice(["Yes", "No"], size=n, p=[0.29, 0.71])
    paperless_billing = rng.choice(["Yes", "No"], size=n, p=[0.59, 0.41])
    payment_method = rng.choice(
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
        size=n,
    )

    base_charge = rng.normal(65, 20, size=n).clip(18, 120)
    monthly_charges = np.round(base_charge, 2)
    total_charges = np.round(monthly_charges * tenure_months + rng.normal(0, 50, size=n), 2)
    total_charges = total_charges.clip(min=0)

    logit = (
        -1.2
        + (-0.045 * tenure_months)
        + np.where(contract == "Month-to-month", 1.1, 0)
        + np.where(contract == "One year", 0.1, 0)
        + np.where(internet_service == "Fiber optic", 0.55, 0)
        + np.where(tech_support == "No", 0.45, 0)
        + np.where(online_security == "No", 0.30, 0)
        + np.where(payment_method == "Electronic check", 0.35, 0)
        + np.where(senior_citizen == 1, 0.25, 0)
        + np.where(partner == "No", 0.15, 0)
        + 0.006 * (monthly_charges - 65)
        + rng.normal(0, 0.6, size=n)
    )
    churn_prob = 1 / (1 + np.exp(-logit))
    churn = (rng.random(n) < churn_prob).astype(int)
    churn_label = np.where(churn == 1, "Yes", "No")

    df = pd.DataFrame(
        {
            "customer_id": customer_id,
            "gender": gender,
            "senior_citizen": senior_citizen,
            "partner": partner,
            "dependents": dependents,
            "tenure_months": tenure_months,
            "contract": contract,
            "internet_service": internet_service,
            "online_security": online_security,
            "tech_support": tech_support,
            "paperless_billing": paperless_billing,
            "payment_method": payment_method,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "churn": churn_label,
        }
    )
    return df


if __name__ == "__main__":
    df = generate_customers()
    out_path = "data/customer_churn.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} rows -> {out_path}")
    print(df["churn"].value_counts(normalize=True).round(3))
