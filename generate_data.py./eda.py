"""
eda.py
------
Exploratory data analysis for the customer churn dataset.
Produces summary statistics in the console and saves charts to
reports/figures/ for use in the README or a slide deck.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
FIG_DIR = "reports/figures"


def load_data(path="data/customer_churn.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def print_summary(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"Rows: {len(df)}  |  Columns: {df.shape[1]}")
    print("\nChurn rate:")
    print(df["churn"].value_counts(normalize=True).round(3))
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    print("\nMissing values:")
    print(missing if not missing.empty else "None")
    print("\nNumeric summary:")
    print(df.describe().round(2))


def plot_churn_distribution(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    df["churn"].value_counts().plot(kind="bar", color=["#4C72B0", "#DD8452"], ax=ax)
    ax.set_title("Customer Churn Distribution")
    ax.set_xlabel("Churn")
    ax.set_ylabel("Number of Customers")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/churn_distribution.png", dpi=150)
    plt.close()


def plot_churn_by_contract(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    churn_by_contract = (
        df.groupby("contract")["churn"].apply(lambda x: (x == "Yes").mean()).sort_values()
    )
    churn_by_contract.plot(kind="barh", color="#55A868", ax=ax)
    ax.set_title("Churn Rate by Contract Type")
    ax.set_xlabel("Churn Rate")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/churn_by_contract.png", dpi=150)
    plt.close()


def plot_tenure_vs_churn(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(
        data=df, x="tenure_months", hue="churn", multiple="stack", bins=24, ax=ax
    )
    ax.set_title("Tenure Distribution by Churn Status")
    ax.set_xlabel("Tenure (months)")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/tenure_vs_churn.png", dpi=150)
    plt.close()


def plot_monthly_charges_vs_churn(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x="churn", y="monthly_charges", ax=ax)
    ax.set_title("Monthly Charges by Churn Status")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/monthly_charges_vs_churn.png", dpi=150)
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    numeric_df = df.select_dtypes(include="number")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Heatmap (Numeric Features)")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/correlation_heatmap.png", dpi=150)
    plt.close()


def main():
    df = load_data()
    print_summary(df)
    plot_churn_distribution(df)
    plot_churn_by_contract(df)
    plot_tenure_vs_churn(df)
    plot_monthly_charges_vs_churn(df)
    plot_correlation_heatmap(df)
    print(f"\nSaved 5 charts to {FIG_DIR}/")


if __name__ == "__main__":
    main()
