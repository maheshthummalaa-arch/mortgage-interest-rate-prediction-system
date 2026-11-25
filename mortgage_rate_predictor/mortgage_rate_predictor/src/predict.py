"""Load the trained model and predict an interest rate + monthly payment."""
import joblib
import pandas as pd
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.joblib"
_model = None


def _get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError("Model not found. Run: python src/train.py")
        _model = joblib.load(MODEL_PATH)
    return _model


def monthly_payment(principal, annual_rate_pct, years):
    r = annual_rate_pct / 100 / 12
    n = years * 12
    if r == 0:
        return principal / n
    return principal * r * (1 + r) ** n / ((1 + r) ** n - 1)


def predict(credit_score, loan_amount, property_value, dti, loan_term, loan_type, market_rate):
    ltv = loan_amount / property_value * 100
    row = pd.DataFrame([{
        "credit_score": credit_score,
        "loan_amount": loan_amount,
        "property_value": property_value,
        "ltv": ltv,
        "dti": dti,
        "loan_term": loan_term,
        "loan_type": loan_type,
        "market_rate": market_rate,
    }])
    rate = float(_get_model().predict(row)[0])
    pay = monthly_payment(loan_amount, rate, int(loan_term))
    return {
        "interest_rate": round(rate, 3),
        "monthly_payment": round(pay, 2),
        "total_paid": round(pay * int(loan_term) * 12, 2),
        "ltv": round(ltv, 2),
    }


if __name__ == "__main__":
    print(predict(720, 300000, 400000, 32, 30, "fixed", 6.5))
