"""Train and compare models, then save the best one."""
import joblib
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "mortgage_data.csv"
MODEL_PATH = ROOT / "models" / "model.joblib"

NUMERIC = ["credit_score", "loan_amount", "property_value", "ltv", "dti", "loan_term", "market_rate"]
CATEGORICAL = ["loan_type"]
TARGET = "interest_rate"


def build_pipeline(model):
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    return Pipeline([("pre", pre), ("model", model)])


def main():
    if not DATA_PATH.exists():
        from generate_data import generate
        DATA_PATH.parent.mkdir(exist_ok=True)
        generate().to_csv(DATA_PATH, index=False)

    df = pd.read_csv(DATA_PATH)
    X, y = df[NUMERIC + CATEGORICAL], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    candidates = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    }

    best_name, best_pipe, best_r2 = None, None, float("-inf")
    for name, model in candidates.items():
        pipe = build_pipeline(model).fit(X_train, y_train)
        pred = pipe.predict(X_test)
        mae = mean_absolute_error(y_test, pred)
        rmse = mean_squared_error(y_test, pred) ** 0.5
        r2 = r2_score(y_test, pred)
        print(f"{name:18s} MAE={mae:.3f}  RMSE={rmse:.3f}  R2={r2:.3f}")
        if r2 > best_r2:
            best_name, best_pipe, best_r2 = name, pipe, r2

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(best_pipe, MODEL_PATH)
    print(f"\nBest model: {best_name} (R2={best_r2:.3f}) -> saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
