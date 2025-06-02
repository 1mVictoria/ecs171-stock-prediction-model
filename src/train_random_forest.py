#!/usr/bin/env python3
"""
Script to train a Random Forest classifier for Buy/Hold/Sell (±2% on next-day return),
using only “today’s” features (no leaked future prices).
"""
import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import pickle

# Hyperparameter distribution for RandomizedSearchCV
PARAM_DIST = {
    'n_estimators': [200, 500],
    'max_depth': [None, 5, 10],
    'min_samples_leaf': [1, 3, 5],
    'max_features': ['sqrt']
}


def load_and_merge(data_dir: str) -> pd.DataFrame:
    """
    Load cleaned price (and—if you want—fundamentals/ESG) and return a single DataFrame.
    Here we assume 'prices_cleaned_scaled.csv' contains columns: symbol, date, close, volume, etc.
    """
    price = pd.read_csv(os.path.join(data_dir, 'prices_cleaned_scaled.csv'))
    # If you also want to merge fundamentals/esg, do so here.
    # fund  = pd.read_csv(os.path.join(data_dir, 'fundamentals_cleaned_scaled.csv'))
    # esg   = pd.read_csv(os.path.join(data_dir, 'esgRisk_cleaned_scaled.csv'))
    # fund = fund.rename(columns={'Ticker Symbol': 'symbol', 'Period Ending': 'date'})
    # esg  = esg.rename(columns={'Symbol': 'symbol'})
    # price['date'] = pd.to_datetime(price['date'])
    # fund['date']  = pd.to_datetime(fund['date'])
    # df = price.merge(fund, on=['symbol','date'], how='left')
    # df = df.merge(esg, on='symbol', how='left')
    price.columns = price.columns.str.lower().str.strip()
    price['date'] = pd.to_datetime(price['date'])
    return price.sort_values(['symbol','date']).reset_index(drop=True)


def create_labels(df: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
    """
    Create Buy/Hold/Sell labels based on next-day return:
      return_next = (close_{t+1} / close_t) - 1
      if > +threshold → 'Buy'
      if < -threshold → 'Sell'
      else              → 'Hold'
    """
    df['return_next'] = df.groupby('symbol')['close'].shift(-1) / df['close'] - 1
    # Drop the final row per symbol where shift(-1) produced NaN
    df = df.dropna(subset=['return_next']).copy()

    def action_classify(r):
        if r > threshold:
            return 'Buy'
        elif r < -threshold:
            return 'Sell'
        else:
            return 'Hold'

    df['label'] = df['return_next'].apply(action_classify)
    # We no longer need return_next afterwards
    df.drop(columns=['return_next'], inplace=True)
    return df


def prepare_features(df: pd.DataFrame) -> (pd.DataFrame, pd.Series):
    """
    Select only “today’s” columns as features. In this minimal example we keep:
      - close
      - volume
      (but you can add any other scaled numeric columns, e.g. fundamentals or ESG)
    """
    # Just as an example, only use 'close' and 'volume' (no fiveday_close)
    X = df[['close', 'volume']].copy()
    y = df['label'].copy()
    return X, y


def main(args):
    # 1) Load price (and optionally fundamentals/ESG)
    df = load_and_merge(args.data_dir)

    # 2) Create next-day labels (no leakage)
    df = create_labels(df, threshold=0.02)

    # 3) Prepare X (only current-day features) and y
    X, y = prepare_features(df)

    # 4) Compute class weights for imbalance
    classes = ['Sell', 'Hold', 'Buy']
    weights = compute_class_weight('balanced', classes=classes, y=y)
    class_weight = dict(zip(classes, weights))
    print('Class weights:', class_weight)

    # 5) Train/test split (stratify by y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.seed
    )

    # 6) Choose classifier
    if args.use_balanced_rf:
        clf = BalancedRandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            max_features=args.max_features or 'sqrt',
            n_jobs=-1, random_state=args.seed,
            replacement=True, sampling_strategy='all'
        )
    else:
        clf = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            max_features=args.max_features or 'sqrt',
            class_weight=class_weight, n_jobs=-1,
            random_state=args.seed
        )

    # 7) Optional hyperparameter search
    if args.random_search:
        tscv = TimeSeriesSplit(n_splits=3)
        rnd_search = RandomizedSearchCV(
            clf, PARAM_DIST, n_iter=args.n_iter,
            cv=tscv, scoring='balanced_accuracy', n_jobs=1, random_state=args.seed
        )
        rnd_search.fit(X_train, y_train)
        clf = rnd_search.best_estimator_
        print('Best params:', rnd_search.best_params_)
    else:
        clf.fit(X_train, y_train)

    # 8) Evaluate
    y_pred = clf.predict(X_test)
    print('=== Classification Report ===')
    print(classification_report(y_test, y_pred))
    print('=== Confusion Matrix ===')
    print(confusion_matrix(y_test, y_pred))

    # 9) Save model
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'RF_model.pkl'), 'wb') as f:
        pickle.dump(clf, f)
    print('Model saved to:', os.path.join(args.output_dir, 'RF_model.pkl'))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir',    type=str,
                   default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_Cleaning', 'cleaned')))
    p.add_argument('--output_dir',  type=str,
                   default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'outputs')))
    p.add_argument('--test_size',   type=float, default=0.2)
    p.add_argument('--n_estimators',type=int, default=200)
    p.add_argument('--max_depth',   type=int, default=None)
    p.add_argument('--min_samples_leaf', type=int, default=3)
    p.add_argument('--max_features',   type=str, default=None)
    p.add_argument('--use_balanced_rf', action='store_true',
                   help='Use BalancedRandomForestClassifier instead of plain RF')
    p.add_argument('--random_search',   action='store_true',
                   help='Perform RandomizedSearchCV over PARAM_DIST')
    p.add_argument('--n_iter',      type=int, default=10)
    p.add_argument('--seed',        type=int, default=42)
    args = p.parse_args()
    main(args)
