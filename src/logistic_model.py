#!/usr/bin/env python3
"""
Multinomial Logistic Regression Pipeline for Predicting Buy / Hold / Sell Signals.
- Loads and merges financial, ESG, and price data
- Engineers technical features (lagged return, moving average, volatility)
- Labels data based on next-period return tertiles
- Trains and evaluates a multinomial logistic regression model
- Reports accuracy, classification metrics, and top feature coefficients
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Load data
price_df = pd.read_csv("data_Cleaning/cleaned/prices_cleaned.csv")
fund_df = pd.read_csv("data_Cleaning/cleaned/fundamentals_cleaned.csv")
esg_df = pd.read_csv("data_Cleaning/cleaned/esgRisk_cleaned.csv")

# Fix column names and formats
fund_df = fund_df.rename(columns={'Ticker Symbol': 'symbol', 'Period Ending': 'date'})
esg_df = esg_df.rename(columns={'Symbol': 'symbol'})
price_df['date'] = pd.to_datetime(price_df['date'])
fund_df['date'] = pd.to_datetime(fund_df['date'])

# Limit ESG to relevant columns
esg_cols = ['symbol', 'Total ESG Risk score', 'Environment Risk Score', 'Social Risk Score', 'Governance Risk Score']
esg_df = esg_df[esg_cols]

# Merge all
df = price_df.merge(fund_df, on=['symbol', 'date'], how='left')
df = df.merge(esg_df, on='symbol', how='left')
df = df.sort_values(['symbol', 'date'])

# Technical features
df['ret_1'] = df.groupby('symbol')['close'].pct_change(1)
df['ret_3'] = df.groupby('symbol')['close'].pct_change(3)
df['ma5'] = df.groupby('symbol')['close'].transform(lambda x: x.rolling(5).mean())
df['std5'] = df.groupby('symbol')['close'].transform(lambda x: x.rolling(5).std())

# Labels (Buy/Hold/Sell)
df['return_next'] = df.groupby('symbol')['close'].shift(-1) / df['close'] - 1
low, high = df['return_next'].quantile([1/3, 2/3])
df['label'] = pd.cut(df['return_next'], bins=[-np.inf, low, high, np.inf], labels=['Sell', 'Hold', 'Buy'])

# Drop rows with missing values in key columns
df = df.dropna(subset=['label', 'ret_1', 'ret_3', 'ma5', 'std5'])

# Select features and labels
drop_cols = ['symbol', 'date', 'close', 'return_next', 'label']
X = df.drop(columns=[col for col in drop_cols if col in df.columns])
X = X.select_dtypes(include=[np.number])
y = df['label']
valid_idx = X.dropna().index
X = X.loc[valid_idx]
y = y.loc[valid_idx]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Logistic regression
model = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='multinomial')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Output metrics
print("=== Classification Report ===")
print(classification_report(y_test, y_pred))
print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))
