import os
import pickle

import pandas as pd
from flask import Flask, render_template, request, flash

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "outputs", "RF_model_binary.pkl")
with open(MODEL_PATH, "rb") as f:
    rf_model = pickle.load(f)

FEAT_PATH = os.path.join(os.path.dirname(__file__), "static", "data", "feature_table.pkl")
feat_df = pd.read_pickle(FEAT_PATH)
# force upper-case on symbol level for case-insensitive look-ups
feat_df.index = feat_df.index.set_levels(
    feat_df.index.levels[0].str.upper(), level=0
)

COL2HTML = {
    "close": "close",
    "volume": "volume",
    "ret_1d": "ret_1d",
    "ret_2d": "ret_2d",
    "momentum_5d": "momentum_5d",
    "ma_5d": "ma_5d",
    "vol_5d": "vol_5d",
    "accel": "accel",
    "rsi_14": "rsi_14",
    "earnings per share": "eps",
    "total revenue": "total_revenue",
    "net income": "net_income",
    "total esg risk score": "total_esg",
    "environment risk score": "env_risk",
    "social risk score": "social_risk",
    "governance risk score": "gov_risk",
}

@app.route("/", methods=["GET", "POST"])
def index():
    prediction   = None
    feature_vals = None   #

    if request.method == "POST":
        ticker = request.form.get("ticker", "").strip().upper()

        if not ticker:
            flash("Please enter a ticker symbol", "danger")
        elif ticker not in feat_df.index.get_level_values(0):
            flash(f"Unknown ticker: {ticker}", "danger")
        else:
            # pull latest row for that ticker
            latest = feat_df.loc[ticker].sort_index().iloc[-1]   # pandas Series

            # prediction
            X      = latest.values.reshape(1, -1)
            label  = rf_model.predict(X)[0]
            prediction = "Action (Buy/Sell)" if label == 1 else "Hold"

            # build {html_id: value} so the template can pre-fill inputs
            feature_vals = {
                COL2HTML[col]: f"{latest[col]:,.4g}"
                for col in latest.index
            }

    return render_template(
        "index.html",
        prediction=prediction,
        feats=feature_vals
    )

@app.errorhandler(404)
def not_found(e):
    return render_template("404.html"), 404

@app.errorhandler(500)
def server_error(e):
    return render_template("500.html"), 500

if __name__ == "__main__":
    app.run(debug=True)
