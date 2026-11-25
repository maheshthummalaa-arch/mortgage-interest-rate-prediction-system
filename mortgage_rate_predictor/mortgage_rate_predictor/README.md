# Mortgage Interest Rate Prediction System

A simple base project that predicts a mortgage interest rate from borrower and loan
details using machine learning (scikit-learn), with a small Flask web interface.

## Structure
```
mortgage_rate_predictor/
├── app.py                 # Flask web app (UI + /api/predict)
├── requirements.txt
├── data/                  # mortgage_data.csv is created here
├── models/                # trained model.joblib is saved here
├── src/
│   ├── generate_data.py   # creates a synthetic dataset
│   ├── train.py           # trains Linear Regression & Random Forest, saves the best
│   └── predict.py         # prediction + monthly payment calculation
└── templates/index.html
```

## Setup & Run
```bash
pip install -r requirements.txt
python src/generate_data.py     # create synthetic data (skip if you have real data)
python src/train.py             # train and save the model
python app.py                   # open http://127.0.0.1:5000
```

## API example
```bash
curl -X POST http://127.0.0.1:5000/api/predict -H "Content-Type: application/json" -d '{
  "credit_score": 720, "loan_amount": 300000, "property_value": 400000,
  "dti": 32, "loan_term": 30, "loan_type": "fixed", "market_rate": 6.5 }'
```

## Using real data
Replace `data/mortgage_data.csv` with your dataset using the same columns:
`credit_score, loan_amount, property_value, ltv, dti, loan_term, loan_type, market_rate, interest_rate`
(or edit `NUMERIC` / `CATEGORICAL` in `src/train.py`), then re-run `python src/train.py`.

## Ideas for extension
- Add features (employment length, region, loan purpose) or fetch live market rates
- Try gradient boosting / XGBoost and hyperparameter tuning
- Add charts, user login, or a database of past predictions

> Note: the bundled dataset is synthetic, for demonstration only. Predictions are not financial advice.
