#!/usr/bin/env python3
"""
Multinomial Logistic Regression Pipeline for Predicting Buy / Hold / Sell Signals.
- Loads and merges financial, ESG, and price data
- Engineers technical features (lagged return, moving average, volatility)
- Labels data based on next-period return tertiles
- Trains and evaluates a multinomial logistic regression model
- Reports accuracy, classification metrics, and top feature coefficients
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

def load_and_merge(data_dir: str) -> pd.DataFrame:
    price = pd.read_csv(os.path.join(data_dir, 'prices_cleaned_scaled.csv'))
    fund  = pd.read_csv(os.path.join(data_dir, 'fundamentals_cleaned_scaled.csv'))
    esg   = pd.read_csv(os.path.join(data_dir, 'esgRisk_cleaned_scaled.csv'))

    fund = fund.rename(columns={'Ticker Symbol': 'symbol', 'Period Ending': 'date'})
    esg  = esg.rename(columns={'Symbol': 'symbol'})
    price['date'] = pd.to_datetime(price['date'])
    fund['date']  = pd.to_datetime(fund['date'])

    df = price.merge(fund, on=['symbol', 'date'], how='left')
    df = df.merge(esg, on='symbol', how='left')
    df.sort_values(['symbol','date'], inplace=True)
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df['ret_1d'] = df.groupby('symbol')['close'].pct_change(1)
    df['ret_3d'] = df.groupby('symbol')['close'].pct_change(3)
    df['ma_5d']  = df.groupby('symbol')['close'].transform(lambda x: x.rolling(5).mean())
    df['vol_5d'] = df.groupby('symbol')['close'].transform(lambda x: x.rolling(5).std())
    return df

def generate_tertile_labels(df: pd.DataFrame) -> pd.DataFrame:
    df['return_next'] = df.groupby('symbol')['close'].shift(-1) / df['close'] - 1
    low, high = df['return_next'].quantile([1/3, 2/3])
    df['label'] = pd.cut(df['return_next'], bins=[-np.inf, low, high, np.inf], labels=[-1, 0, 1])
    return df.dropna(subset=['label', 'ret_1d', 'ret_3d', 'ma_5d', 'vol_5d'])

def prepare_data(df: pd.DataFrame):
    drop_cols = ['symbol','date','close','return_next','label']
    X = df.drop(columns=[col for col in drop_cols if col in df.columns])
    X = X.select_dtypes(include=[np.number])
    y = df['label'].astype(int)
    valid_idx = X.dropna().index
    return X.loc[valid_idx], y.loc[valid_idx]

def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, solver='lbfgs')
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))

    # Report as DataFrame
    report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True, zero_division=0)).transpose()
    report_df.rename(index={"-1": "Sell (-1)", "0": "Hold (0)", "1": "Buy (1)"}, inplace=True)

    plt.figure(figsize=(8, 5))
    plt.table(cellText=report_df.round(2).values,
              rowLabels=report_df.index,
              colLabels=report_df.columns,
              loc='center', cellLoc='center')
    plt.title("Multinomial Logistic Regression Report")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("logreg_classification_report.png")
    plt.show()

    # Prediction distribution
    weighted_score = model.predict_proba(X_test_scaled).dot([1, 0, -1])
    plt.hist(weighted_score, bins=50)
    plt.title("Predicted Class Confidence Distribution")
    plt.xlabel("Weighted Prediction Score")
    plt.ylabel("Frequency")
    plt.show()

    # Feature importance
    print("=== Top Features (absolute coefficients) ===")
    coef_df = pd.DataFrame(model.coef_, columns=X.columns, index=['Class -1', 'Class 0', 'Class 1']).T
    coef_df['MaxAbs'] = coef_df.abs().max(axis=1)
    print(coef_df.sort_values(by='MaxAbs', ascending=False).head(10))

def main():
    df = load_and_merge("./data_Cleaning/cleaned")  
    df = engineer_features(df)
    df = generate_tertile_labels(df)
    X, y = prepare_data(df)
    train_and_evaluate(X, y)

if __name__ == '__main__':
    main()
