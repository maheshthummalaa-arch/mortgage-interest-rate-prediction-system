"""Flask web app for the Mortgage Interest Rate Prediction System."""
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from predict import predict  # noqa: E402

app = Flask(__name__)


def parse(data):
    return dict(
        credit_score=int(data["credit_score"]),
        loan_amount=float(data["loan_amount"]),
        property_value=float(data["property_value"]),
        dti=float(data["dti"]),
        loan_term=int(data["loan_term"]),
        loan_type=data["loan_type"],
        market_rate=float(data["market_rate"]),
    )


@app.route("/", methods=["GET", "POST"])
def index():
    result, error, form = None, None, request.form
    if request.method == "POST":
        try:
            args = parse(request.form)
            if args["property_value"] <= 0 or args["loan_amount"] <= 0:
                raise ValueError("Loan amount and property value must be positive.")
            result = predict(**args)
        except Exception as e:
            error = str(e)
    return render_template("index.html", result=result, error=error, form=form)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        return jsonify(predict(**parse(request.get_json(force=True))))
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
