"""Generate a synthetic mortgage dataset.

Replace data/mortgage_data.csv with real data (same columns) when you have it.
"""
import numpy as np
import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "mortgage_data.csv"


def generate(n=5000, seed=42):
    rng = np.random.default_rng(seed)

    credit_score = rng.integers(580, 851, n)
    loan_amount = rng.integers(50_000, 800_001, n)
    property_value = (loan_amount / rng.uniform(0.5, 0.97, n)).astype(int)
    ltv = loan_amount / property_value * 100                 # loan-to-value %
    dti = rng.uniform(10, 50, n)                              # debt-to-income %
    loan_term = rng.choice([15, 20, 30], n, p=[0.25, 0.1, 0.65])
    loan_type = rng.choice(["fixed", "adjustable"], n, p=[0.8, 0.2])
    market_rate = rng.uniform(3.0, 7.5, n)                    # benchmark rate %

    rate = (
        market_rate
        + (750 - credit_score) * 0.006
        + (ltv - 80) * 0.02
        + (dti - 30) * 0.015
        + np.where(loan_term == 30, 0.25, np.where(loan_term == 20, 0.1, -0.2))
        + np.where(loan_type == "adjustable", -0.35, 0.0)
        + rng.normal(0, 0.15, n)
    ).clip(2.0, 12.0)

    return pd.DataFrame({
        "credit_score": credit_score,
        "loan_amount": loan_amount,
        "property_value": property_value,
        "ltv": ltv.round(2),
        "dti": dti.round(2),
        "loan_term": loan_term,
        "loan_type": loan_type,
        "market_rate": market_rate.round(2),
        "interest_rate": rate.round(3),
    })


if __name__ == "__main__":
    df = generate()
    DATA_PATH.parent.mkdir(exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"Saved {len(df)} rows to {DATA_PATH}")
