#!/usr/bin/env python3
"""
Script to train a Random Forest (or Balanced Random Forest) classifier for Buy/Hold/Sell signals,
apply risk-management, and include technical feature engineering to boost accuracy.
- Adds lagged returns and rolling statistics (MA, volatility)
- Uses tertile labeling by default
- Option for BalancedRandomForestClassifier
- Hyperparameter search with RandomizedSearchCV
- Prints top-10 feature importances
"""
import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pickle

# Hyperparameter distribution for RandomizedSearchCV
PARAM_DIST = {
    'n_estimators': [500, 1000],
    'max_depth': [None, 5, 10],
    'min_samples_leaf': [1, 3, 5],
    'max_features': ['sqrt', 0.3, 0.5]
}

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
    df = df.sort_values(['symbol','date'])

    # Technical features
    df['ret_1'] = df.groupby('symbol')['close'].pct_change(1)
    df['ret_3'] = df.groupby('symbol')['close'].pct_change(3)
    df['ma5']   = df.groupby('symbol')['close'].transform(lambda x: x.rolling(5).mean())
    df['std5']  = df.groupby('symbol')['close'].transform(lambda x: x.rolling(5).std())

    return df

def create_labels(df: pd.DataFrame) -> pd.DataFrame:
    # Tertile-based labeling for balanced classes
    df['return_next'] = df.groupby('symbol')['close'].shift(-1) / df['close'] - 1
    low, high = df['return_next'].quantile([1/3, 2/3])
    bins = [-np.inf, low, high, np.inf]
    df['label'] = pd.cut(df['return_next'], bins=bins, labels=['Sell','Hold','Buy'])
    return df.dropna(subset=['label','ret_1','ret_3','ma5','std5'])

def prepare_features(df: pd.DataFrame):
    drop_cols = ['symbol','date','close','return_next','label']
    X = df.drop(columns=drop_cols)
    X_num = X.select_dtypes(include=[np.number]).copy()
    y = df['label']
    valid_idx = X_num.dropna().index
    return X_num.loc[valid_idx], y.loc[valid_idx]

def main(args):
    df = load_and_merge(args.data_dir)
    df = create_labels(df)
    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size,
        stratify=y, random_state=args.seed)

    # Choose classifier
    if args.use_balanced_rf:
        clf_cls = BalancedRandomForestClassifier
        clf_params = dict(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            max_features=args.max_features or 'sqrt',
            n_jobs=-1, random_state=args.seed,
            replacement=True, sampling_strategy='all'
        )
    else:
        clf_cls = RandomForestClassifier
        clf_params = dict(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            max_features=args.max_features or 'sqrt',
            class_weight='balanced', n_jobs=-1,
            random_state=args.seed
        )
    base_clf = clf_cls(**clf_params)

    # Hyperparameter search
    if args.random_search:
        tscv = TimeSeriesSplit(n_splits=5)
        search = RandomizedSearchCV(
            base_clf, PARAM_DIST, n_iter=args.n_iter,
            cv=tscv, scoring='balanced_accuracy',
            n_jobs=-1, random_state=args.seed
        )
        search.fit(X_train, y_train)
        clf = search.best_estimator_
        print('Best params:', search.best_params_)
    else:
        base_clf.fit(X_train, y_train)
        clf = base_clf

    # Evaluate
    y_pred = clf.predict(X_test)
    print('=== Classification Report ===')
    print(classification_report(y_test, y_pred))
    print('=== Confusion Matrix ===')
    print(confusion_matrix(y_test, y_pred))

    # Feature importances
    feat_imp = pd.Series(clf.feature_importances_, index=X.columns)
    print('=== Top 10 Features ===')
    print(feat_imp.sort_values(ascending=False).head(10))

    # Position sizing for Buy signals
    if args.portfolio_value:
        df_test = df.loc[X_test.index].copy()
        df_test['prediction'] = y_pred
        buys = df_test[df_test['prediction']=='Buy']
        buys.loc[:, 'stop_loss'] = buys['close'] * (1 - args.stop_loss_pct)
        buys.loc[:, 'risk_per_share'] = buys['close'] - buys['stop_loss']
        buys.loc[:, 'position_size'] = (args.portfolio_value * args.risk_pct) / buys['risk_per_share']
        print('=== Buy Recommendations (first 10) ===')
        print(buys[['symbol','date','close','stop_loss','position_size']].head(10))

    # Save model
    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(args.output_dir, 'model.pkl')
    with open(path, 'wb') as f:
        pickle.dump(clf, f)
    print(f'Model saved to {path}')

if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=str,
                   default=os.path.abspath(os.path.join(os.path.dirname(__file__),'..','data_Cleaning','cleaned')))
    p.add_argument('--output_dir', type=str,
                   default=os.path.abspath(os.path.join(os.path.dirname(__file__),'..','outputs')))
    p.add_argument('--test_size', type=float, default=0.2)
    p.add_argument('--n_estimators', type=int, default=500)
    p.add_argument('--max_depth', type=int, default=None)
    p.add_argument('--min_samples_leaf', type=int, default=3)
    p.add_argument('--max_features', type=str, default=None)
    p.add_argument('--use_balanced_rf', action='store_true',
                   help='Use BalancedRandomForestClassifier')
    p.add_argument('--random_search', action='store_true')
    p.add_argument('--n_iter', type=int, default=30)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--portfolio_value', type=float, default=0,
                   help='Total portfolio value for sizing')
    p.add_argument('--risk_pct', type=float, default=0.01,
                   help='Risk per trade fraction')
    p.add_argument('--stop_loss_pct', type=float, default=0.01,
                   help='Stop-loss fraction')
    args = p.parse_args()
    main(args)
