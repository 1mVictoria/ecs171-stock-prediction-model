#!/usr/bin/env python3
"""
Train a Random-Forest (or Balanced RF) to decide
***“should I take action tomorrow?”*** on the basis of today's features.

Binary labels:

    1 → actionable  (Buy **or** Sell; ±threshold on next-day return)
    0 → don't act   (Hold; within ±threshold)

Prerequisites

- Python 3.8+  
- pandas 1.5.3  
- numpy 1.24.3  
- scikit-learn 1.2.2  
- imbalanced-learn 0.10.1  
- matplotlib, seaborn  
"""
import os, argparse, pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns


# ──────────────────────────────────────────────────────────────
# Hyper-parameter grid for optional RandomizedSearchCV
PARAM_DIST = {
    "n_estimators":     [200, 500],
    "max_depth":        [None, 5, 10],
    "min_samples_leaf": [1, 3, 5],
    "max_features":     ["sqrt"],
}
# ──────────────────────────────────────────────────────────────
DEFAULT_HORIZON = 21

def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Relative Strength Index (RSI) computed with Wilder's smoothing.
    Returns a Series of RSI values aligned with *series*.
    """
    delta = series.diff()
    gain = delta.clip(lower=0).fillna(0)
    loss = (-delta.clip(upper=0)).fillna(0)
    # Wilder's smoothing
    gain = gain.ewm(alpha=1/window, adjust=False).mean()
    loss = loss.ewm(alpha=1/window, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


import os
import pandas as pd

def load_and_merge(data_dir: str) -> pd.DataFrame:
    """
    1) Reads prices_cleaned.csv (has columns: symbol, date, close, volume, …).
    2) Reads fundamentals_cleaned.csv (has columns exactly as in your file).
    3) Reads esgRisk_cleaned.csv (has columns exactly as in your file).

    Returns a single DataFrame where each price row carries all the columns
    from the fundamentals and ESG files, joined purely on 'symbol'.
    """
    # --- 1) Load price data (daily time series) ---
    price = pd.read_csv(
        os.path.join(data_dir, "prices_cleaned.csv"),
        parse_dates=["date"]
    )
    # Lowercase + strip whitespace from headers
    price.columns = price.columns.str.lower().str.strip()
    price = price.sort_values(["symbol", "date"]).reset_index(drop=True)

    # --- 2) Load fundamentals (static per symbol) ---
    fund = pd.read_csv(os.path.join(data_dir, "fundamentals_cleaned.csv"))
    fund.columns = fund.columns.str.lower().str.strip()
    # Rename 'ticker symbol' → 'symbol' so we can join on that key
    fund = fund.rename(columns={"ticker symbol": "symbol"})
    # Strip whitespace in the symbol column itself
    fund["symbol"] = fund["symbol"].astype(str).str.strip()

    # --- 3) Load ESG (static per symbol) ---
    esg = pd.read_csv(os.path.join(data_dir, "esgRisk_cleaned.csv"))
    esg.columns = esg.columns.str.lower().str.strip()
    # The ESG file has a 'symbol' column already (after lowercasing).
    esg["symbol"] = esg["symbol"].astype(str).str.strip()

    # --- 4) Merge price ← fundamentals on "symbol" (left join) ---
    merged = pd.merge(
        price,
        fund,
        on="symbol",
        how="left"
    )

    # --- 5) Merge merged ← ESG on "symbol" (left join) ---
    merged = pd.merge(
        merged,
        esg,
        on="symbol",
        how="left"
    )

    return merged



def create_labels(df: pd.DataFrame,
                  threshold: float = 0.01,
                  horizon: int = DEFAULT_HORIZON) -> pd.DataFrame:
    """
    Binary label creation over *horizon* steps ahead (default ≈ 1 month).

        return_h = (close_{t+h} / close_t) - 1

        if |return_h|  > threshold  →  1   (actionable)
        else                         →  0   (don't act)

    • horizon = number of rows to look ahead **per symbol**  
      (21 ≈ one trading month; 5 would be one trading week, etc.)
    • threshold default bumped to ±5 % because monthly moves are larger.
    """
    df["return_h"] = (
        df.groupby("symbol")["close"].shift(-horizon) / df["close"] - 1
    )
    df = df.dropna(subset=["return_h"]).copy()

    df["label"] = (df["return_h"].abs() > threshold).astype(int)
    df.drop(columns=["return_h"], inplace=True)
    return df


def prepare_features(df: pd.DataFrame):
    """
    From the merged df (price + fundamentals + ESG), produce:
      • 9 technical features (close, volume, ret_1d, …, rsi_14)
      • 3 fundamental features:
          - earnings per share
          - total revenue
          - net income
      • 4 ESG features:
          - total esg risk score
          - environment risk score
          - social risk score
          - governance risk score

    Returns X (feature DataFrame) and y (Series of labels).
    """
    df = df.copy()
    grp_close = df.groupby("symbol")["close"]

    # --- 1) Create technical features exactly as before ---
    df["ret_1d"]      = grp_close.pct_change(1)
    df["ret_2d"]      = grp_close.pct_change(2)
    df["momentum_5d"] = grp_close.transform(lambda x: x / x.shift(5))
    df["ma_5d"]       = grp_close.transform(lambda x: x.rolling(5).mean())
    df["vol_5d"]      = grp_close.transform(lambda x: x.rolling(5).std())
    df["accel"]       = grp_close.diff().diff()
    df["rsi_14"]      = grp_close.transform(compute_rsi)

    # --- 2) Fundamental features (use exact lowercased names) ---
    #    If you want to include more fundamentals, add them here:
    df["earnings per share"] = df["earnings per share"]
    df["total revenue"]      = df["total revenue"]
    df["net income"]         = df["net income"]

    # --- 3) ESG features (use exact lowercased names) ---
    df["total esg risk score"]    = df["total esg risk score"]
    df["environment risk score"]  = df["environment risk score"]
    df["social risk score"]       = df["social risk score"]
    df["governance risk score"]   = df["governance risk score"]
    # If you want to include controversy score, you can add:
    # df["controversy score"] = df["controversy score"]

    # --- 4) Compile the complete list of features (9 + 3 + 4 = 16) ---
    feature_cols = [
        # 9 technical features:
        "close", "volume",
        "ret_1d", "ret_2d", "momentum_5d",
        "ma_5d", "vol_5d", "accel", "rsi_14",

        # 3 fundamentals:
        "earnings per share",
        "total revenue",
        "net income",

        # 4 ESG features:
        "total esg risk score",
        "environment risk score",
        "social risk score",
        "governance risk score"
        # If using controversy score: add "controversy score" here
    ]

    # --- 5) Drop rows missing any of these features or the label ---
    df = df.dropna(subset=feature_cols + ["label"]).copy()

    X = df[feature_cols]
    y = df["label"]
    return X, y



def main(args):
    # 1) Load & label
    df = load_and_merge(args.data_dir)
    df = create_labels(df,
                   threshold=args.threshold,
                   horizon=args.horizon)

    # 2) Features / target
    X, y = prepare_features(df)

    # 3) Class weights (binary)
    classes = [0, 1]
    weights = compute_class_weight("balanced", classes=classes, y=y)
    class_weight = dict(zip(classes, weights))
    print("Class weights:", class_weight)

    # 4) Train/test split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.seed
    )

    # 5) Choose RF variant
    if args.use_balanced_rf:
        clf = BalancedRandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            max_features=args.max_features or "sqrt",
            n_jobs=-1,
            random_state=args.seed,
            replacement=True,
            sampling_strategy="all",
        )
    else:
        clf = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            max_features=args.max_features or "sqrt",
            class_weight=class_weight,
            n_jobs=-1,
            random_state=args.seed,
        )
    

    # 6) Optional random search
    if args.random_search:
        tscv = TimeSeriesSplit(n_splits=3)
        search = RandomizedSearchCV(
            clf,
            PARAM_DIST,
            n_iter=args.n_iter,
            cv=tscv,
            scoring="balanced_accuracy",
            n_jobs=1,
            random_state=args.seed,
        )
        search.fit(X_tr, y_tr)
        clf = search.best_estimator_
        print("Best params:", search.best_params_)
    else:
        clf.fit(X_tr, y_tr)
    # 7.5) Top‑10 feature importances
    importances = (
        pd.Series(clf.feature_importances_, index=X_tr.columns)
          .sort_values(ascending=False)
    )
    top_k = min(10, len(importances))
    print(f"=== Top {top_k} Feature Importances ===")
    print(importances.head(top_k).to_string(float_format="%.4f"))
    top_k = 10
    print("\nTop features (k = {}):".format(min(top_k, len(importances))))
    print(importances.head(top_k).to_string(float_format="%.4f"))
    # 7) Evaluation
    y_pred = clf.predict(X_te)
    print("=== Classification Report (0 = Hold, 1 = Action) ===")
    print(classification_report(y_te, y_pred, target_names=["Hold", "Action"]))
    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_te, y_pred))
    
    plots_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(plots_dir, exist_ok=True)
    # --- (a) Feature-importance bar chart -----------------------
    plt.figure(figsize=(6, 4))
    importances.iloc[::-1].plot(kind="barh")          # reverse for top-to-bottom
    plt.xlabel("Gini importance")
    plt.title("Top 9 Feature Importances")
    plt.tight_layout()
    feat_path = os.path.join(plots_dir, "rf_feature_importance.pdf")
    plt.savefig(feat_path)
    plt.close()

    # --- (b) Confusion-matrix heatmap ---------------------------
    cm = confusion_matrix(y_te, y_pred)
    plt.figure(figsize=(4, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Pred Hold", "Pred Action"],
        yticklabels=["True Hold", "True Action"],
    )
    plt.title("Confusion Matrix")
    plt.tight_layout()
    cm_path = os.path.join(plots_dir, "rf_confusion_matrix.pdf")
    plt.savefig(cm_path)
    plt.close()

    print("Figures saved to:", plots_dir)

    # 8) Save model
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "RF_model_binary.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(clf, f)
    print("Model saved to:", out_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data_dir",
        type=str,
        default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "data_Cleaning", "cleaned")
        ),
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs")),
    )
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--n_estimators", type=int, default=200)
    p.add_argument("--max_depth", type=int, default=None)
    p.add_argument("--min_samples_leaf", type=int, default=3)
    p.add_argument("--max_features", type=str, default=None)
    p.add_argument(
        "--use_balanced_rf",
        action="store_true",
        help="Use BalancedRandomForestClassifier instead of plain RF",
    )
    p.add_argument(
        "--random_search",
        action="store_true",
        help="Perform RandomizedSearchCV over PARAM_DIST",
    )
    p.add_argument("--horizon", type=int, default=DEFAULT_HORIZON,
               help="Look-ahead steps per symbol (≈21 for a month, 5 for a week)")
    p.add_argument("--threshold", type=float, default=0.01,
                help="±% move considered actionable (e.g. 0.05 = 5 %)")

    p.add_argument("--n_iter", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    main(args)
