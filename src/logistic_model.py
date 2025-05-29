#!/usr/bin/env python3
"""
Logistic Regression Pipeline for Predicting Buy (1) vs Hold/Sell (0) Signals.
- Loads and merges financial, ESG, and price data
- Engineers technical features (lagged return, moving average, volatility)
- Labels data based on next-period return direction
- Trains and evaluates a logistic regression model
- Reports accuracy, classification metrics, and top features
"""

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def import_and_merge_datasets(data_dir: str) -> pd.DataFrame:
    """Load and merge price, fundamental, and ESG datasets."""
    price_df = pd.read_csv(os.path.join(data_dir, 'prices_cleaned_scaled.csv'))
    fund_df = pd.read_csv(os.path.join(data_dir, 'fundamentals_cleaned_scaled.csv'))
    esg_df = pd.read_csv(os.path.join(data_dir, 'esgRisk_cleaned_scaled.csv'))

    fund_df = fund_df.rename(columns={'Ticker Symbol': 'symbol', 'Period Ending': 'date'})
    esg_df = esg_df.rename(columns={'Symbol': 'symbol'})
    price_df['date'] = pd.to_datetime(price_df['date'])
    fund_df['date'] = pd.to_datetime(fund_df['date'])

    merged = price_df.merge(fund_df, on=['symbol', 'date'], how='left')
    merged = merged.merge(esg_df, on='symbol', how='left')
    merged.sort_values(['symbol', 'date'], inplace=True)

    return merged


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators: lagged returns, moving average, rolling std dev."""
    df['ret_1d'] = df.groupby('symbol')['close'].pct_change(1)
    df['ma_5d'] = df.groupby('symbol')['close'].transform(lambda x: x.rolling(5).mean())
    df['vol_5d'] = df.groupby('symbol')['close'].transform(lambda x: x.rolling(5).std())
    return df


def generate_binary_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Generate binary classification labels: 1 = Buy if next return > 0, else 0."""
    df['return_next'] = df.groupby('symbol')['close'].shift(-1) / df['close'] - 1
    df['label'] = (df['return_next'] > 0).astype(int)
    return df.dropna(subset=['label', 'ret_1d', 'ma_5d', 'vol_5d'])


def select_features_and_labels(df: pd.DataFrame):
    """Prepare X (features) and y (labels), filtering for numeric data only."""
    drop_columns = ['symbol', 'date', 'close', 'return_next', 'label']
    X = df.drop(columns=[col for col in drop_columns if col in df.columns])
    X = X.select_dtypes(include=[np.number])
    y = df['label']
    valid_index = X.dropna().index
    return X.loc[valid_index], y.loc[valid_index]


def run_logistic_training(X, y, test_size=0.2, seed=42):
    """Train logistic regression and return metrics."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    model = LogisticRegression(max_iter=1000, solver='lbfgs')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("=== Accuracy ===")
    print(f"{accuracy_score(y_test, y_pred):.4f}")
    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred))
    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))

    print("=== Top Feature Coefficients ===")
    coefs = pd.Series(model.coef_[0], index=X.columns)
    print(coefs.sort_values(key=abs, ascending=False).head(10))

    return model


def main(args):
    df = import_and_merge_datasets(args.data_dir)
    df = engineer_features(df)
    df = generate_binary_labels(df)
    X, y = select_features_and_labels(df)
    run_logistic_training(X, y, test_size=args.test_size, seed=args.seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_Cleaning', 'cleaned')))
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    main(args)
