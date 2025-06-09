#!/usr/bin/env python3
"""
Logistic Regression Pipeline for Predicting Action (1) vs Hold (0) Signals.
- Loads and merges financial, ESG, and price data
- Engineers technical features (lagged return, moving average, volatility)
- Labels data based on next-period return threshold
- Trains and evaluates a logistic regression model
- Reports accuracy, classification metrics, and saves confusion matrix PDF
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# --- 1. Load and merge datasets ---
price_df = pd.read_csv("data_Cleaning/cleaned/prices_cleaned.csv")
fund_df  = pd.read_csv("data_Cleaning/cleaned/fundamentals_cleaned.csv")
esg_df   = pd.read_csv("data_Cleaning/cleaned/esgRisk_cleaned.csv")

fund_df = fund_df.rename(columns={'Ticker Symbol': 'symbol', 'Period Ending': 'date'})
esg_df  = esg_df.rename(columns={'Symbol': 'symbol'})
price_df['date'] = pd.to_datetime(price_df['date'])
fund_df['date']  = pd.to_datetime(fund_df['date'])

esg_cols = [
    'symbol',
    'Total ESG Risk score',
    'Environment Risk Score',
    'Social Risk Score',
    'Governance Risk Score'
]
esg_df = esg_df[esg_cols]

df = price_df.merge(fund_df, on=['symbol','date'], how='left')
df = df.merge(esg_df, on='symbol', how='left')
df = df.sort_values(['symbol','date'])

# --- 2. Engineer technical features ---
df['ret_1'] = df.groupby('symbol')['close'].pct_change(1)
df['ret_3'] = df.groupby('symbol')['close'].pct_change(3)
df['ma5']   = df.groupby('symbol')['close'].transform(lambda x: x.rolling(5).mean())
df['std5']  = df.groupby('symbol')['close'].transform(lambda x: x.rolling(5).std())

# --- 3. Generate binary labels: Action vs Hold ---
df['return_next'] = df.groupby('symbol')['close'].shift(-1) / df['close'] - 1
df['label'] = (df['return_next'].abs() > 0.01).astype(int)  # 1 = Action, 0 = Hold

# Only drop rows where the four technical features or label are missing
df = df.dropna(subset=['label','ret_1','ret_3','ma5','std5'])

# --- 4. Prepare features (X) and labels (y) ---
drop_cols = ['symbol','date','close','return_next','label']
X = df.drop(columns=drop_cols)
X = X.select_dtypes(include=[np.number])
y = df['label'].loc[X.index]

# --- Impute any remaining NaNs in X (e.g., ESG or fundamentals) ---
imp = SimpleImputer(strategy='median')
X = pd.DataFrame(
    imp.fit_transform(X),
    columns=X.columns,
    index=X.index
)

# --- 5. Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# --- 6. Train logistic regression ---
model = LogisticRegression(
    max_iter=1000,
    solver='lbfgs',
    class_weight='balanced'
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# --- 7. Print metrics to console ---
print("=== Accuracy ===")
print(f"{model.score(X_test, y_test):.4f}")
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, zero_division=0))
print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

# --- 8. Generate, save confusion matrix as PDF ---
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
disp = ConfusionMatrixDisplay(cm, display_labels=['Hold', 'Action'])

fig, ax = plt.subplots(figsize=(4, 4))
disp.plot(ax=ax, cmap='Blues', colorbar=False, values_format='d')
ax.set_title("Logistic Regression\nConfusion Matrix")
fig.tight_layout()

fig.savefig("logreg_cm.pdf", bbox_inches="tight")
plt.close(fig)
print(f"Saved confusion matrix to: {os.path.abspath('logreg_cm.pdf')}")

# --- 9. Print top feature coefficients ---
print("\n=== Top Feature Coefficients (abs value) ===")
coefs = pd.Series(model.coef_[0], index=X.columns)
print(coefs.abs().sort_values(ascending=False).head(10))
