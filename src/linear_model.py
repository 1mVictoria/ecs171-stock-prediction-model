"""
Train a Linear Regression model to decide
***“should I take action tomorrow?”*** on the basis of today's features.

Binary labels:

    1 → actionable  (Buy **or** Sell; ±threshold on next-day return)
    0 → don't act   (Hold; within ±threshold)

No future prices are leaked.

"""
import os
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

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
    df["ret_1d"] = grp_close.pct_change(1)
    df["ret_2d"] = grp_close.pct_change(2)
    df["momentum_5d"] = grp_close.transform(lambda x: x / x.shift(5))
    df["ma_5d"] = grp_close.transform(lambda x: x.rolling(5).mean())
    df["vol_5d"] = grp_close.transform(lambda x: x.rolling(5).std())
    df["accel"] = grp_close.diff().diff()
    df["rsi_14"] = grp_close.transform(compute_rsi)

    # --- 2) Fundamental features (use exact lowercased names) ---
    #    If you want to include more fundamentals, add them here:
    df["earnings per share"] = df["earnings per share"]
    df["total revenue"] = df["total revenue"]
    df["net income"] = df["net income"]

    # --- 3) ESG features (use exact lowercased names) ---
    df["total esg risk score"] = df["total esg risk score"]
    df["environment risk score"] = df["environment risk score"]
    df["social risk score"] = df["social risk score"]
    df["governance risk score"] = df["governance risk score"]
    # If you want to include controversy score, you can add:
    # df["controversy score"] = df["controversy score"]

    # --- 4) Compile the complete list of features (9 + 3 + 4 = 16) ---
    feature_cols = [
        # 9 technical features:
        # "close", "volume",
        "ret_1d", "momentum_5d",
        "ma_5d", "rsi_14",

        # 3 fundamentals:
        # "earnings per share",
        # "total revenue",
        "net income",

        # 4 ESG features:
        "total esg risk score",
        # "environment risk score",
        # "social risk score",
        # "governance risk score"
        # If using controversy score: add "controversy score" here
    ]

    # --- 5) Drop rows missing any of these features or the label ---
    df = df.dropna(subset=feature_cols + ["label"]).copy()

    X = df[feature_cols]
    y = df["label"]
    return X, y

# Classify binary actions
def action_classify(val, threshold= 0.01):
    # 1 -> action; 0 -> Hold
    return int(abs(val) > threshold)

class LinearRegression32GD:
    def __init__(self):
        self.theta = None

    def fit(self, X_train, y_train, X_test, y_test, lr=0.0001, epochs=100, batch_size=32):
        self.theta = np.random.randn(X_train.shape[1])
        m = X_train.shape[0]
        if batch_size is None:
            batch_size = m

        # Create empty lists for recording the MSE scores of
        # the linear regression model on training data & test data
        mse_train_history = []
        mse_test_history = []

        # Iterate over the batches in each epoch
        for epoch in range(epochs):
            # This line generates random indices from 0 to m (the total number of samples in the training set) without replacement.
            # m is the size of the training dataset. The number of indices generated is equal to the batch_size.
            indices = np.random.choice(m, batch_size)
            # creates a batch of input features for training
            X_batch = X_train[indices]
            # selects the corresponding subset of y values
            y_batch = y_train[indices]

            # generating the predictions on X_batch
            y_pred = X_batch.dot(self.theta)
            # compute the gradient of the loss function
            error = y_pred - y_batch
            gradient = 2 / batch_size * X_batch.T.dot(error)
            # updating the weights with the computed gradient
            self.theta -= lr * gradient

            # Predict
            y_train_pred = X_train.dot(self.theta)
            y_test_pred = X_test.dot(self.theta)

            # Record the MSE scores
            mse_train_history.append(mean_squared_error(y_train, y_train_pred))
            mse_test_history.append(mean_squared_error(y_test, y_test_pred))

        return mse_train_history, mse_test_history



def main(args):
    # Merge & Load cleaned dataset
    df = load_and_merge(args.data_dir)
    df = create_labels(df, threshold=args.threshold, horizon=args.horizon)

    # Select features and labels
    X, y = prepare_features(df)

    # Split dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=args.test_size, random_state=42)

    # Train using 32-batch-size GD
    model = LinearRegression32GD()
    mse_train, mse_test = model.fit(X_train, y_train, X_test, y_test,
                                    lr=args.lr, epochs=args.epochs, batch_size=args.batch_size)

    # Plot MSE
    plt.plot(mse_train, label='Train MSE')
    plt.plot(mse_test, label='Test MSE')
    plt.legend()
    plt.title("Train/Test MSE over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.show()

    # Classification report
    y_pred = X_test.dot(model.theta)
    y_pred_class = [action_classify(val, threshold=args.threshold) for val in y_pred]

    # Transfer to DataFrame for plotting
    report_dict = classification_report(
        y_test, y_pred_class, output_dict=True, target_names=["Hold", "Action"]
    )
    report_df = pd.DataFrame(report_dict).transpose()

    print("=== Classification Report ===")
    print(report_df.round(2))
    # Plot Classification Report
    plt.figure(figsize=(8, 6))
    plt.table(cellText=report_df.round(2).values,
              rowLabels=report_df.index,
              colLabels=report_df.columns,
              cellLoc='center',
              loc='center')
    plt.axis('off')
    plt.title("Linear Regression Model Report")
    plt.tight_layout()
    plt.show()

    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred_class))

    # Draw & Save Confusion-matrix heatmap
    cm = confusion_matrix(y_test, y_pred_class)
    plt.figure(figsize=(4, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Pred Hold", "Pred Action"],
        yticklabels=["True Hold", "True Action"]
    )
    plt.title("Linear Regression Confusion Matrix")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    # Parse command-line arguments for training the linear regression model (uncomment for using)
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../data_Cleaning/cleaned")
    parser.add_argument("--threshold", type=float, default=0.01)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    main(args)

