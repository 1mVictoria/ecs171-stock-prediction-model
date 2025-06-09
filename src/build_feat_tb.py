#!/usr/bin/env python3
"""
build_feat_tb.py  – Pre-compute the full feature table and pickle it
                   for the Flask web-app.

• Reads the three cleaned CSVs (prices, fundamentals, ESG)
• Writes to  web-interface/static/data/feature_table.pkl
"""

import os
import pickle
import pandas as pd

# reusing from train_random_forest.py
from train_random_forest import load_and_merge, compute_rsi

FEATURE_COLS = [
    # 9 technical
    "close", "volume",
    "ret_1d", "ret_2d", "momentum_5d",
    "ma_5d", "vol_5d", "accel", "rsi_14",
    # 3 fundamental
    "earnings per share", "total revenue", "net income",
    # 4 ESG
    "total esg risk score", "environment risk score",
    "social risk score", "governance risk score",
]



def build_feature_table():
    base_dir = os.path.dirname(__file__)                     # …/src
    data_dir = os.path.abspath(os.path.join(base_dir, "..", "data_Cleaning", "cleaned"))

    print(f"Loading & merging data from: {data_dir}")
    df = load_and_merge(data_dir)                            # symbol, date, …

    # 
    grp = df.groupby("symbol")["close"]
    df["ret_1d"]      = grp.pct_change(1)
    df["ret_2d"]      = grp.pct_change(2)
    df["momentum_5d"] = grp.transform(lambda x: x / x.shift(5))
    df["ma_5d"]       = grp.transform(lambda x: x.rolling(5).mean())
    df["vol_5d"]      = grp.transform(lambda x: x.rolling(5).std())
    df["accel"]       = grp.diff().diff()
    df["rsi_14"]      = grp.transform(compute_rsi)

    # Assemble final table
    df["symbol"] = df["symbol"].str.upper()  
    out_df = (
        df[["symbol", "date"] + FEATURE_COLS]
        .dropna(subset=FEATURE_COLS)   # ← keep rows even if other feats NaN
        .set_index(["symbol", "date"])
        .sort_index()
    )

    project_root = os.path.abspath(os.path.join(base_dir, ".."))
    out_path     = os.path.join(
        project_root, "web-interface", "static", "data", "feature_table.pkl"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "wb") as f:
        pickle.dump(out_df, f)

    print(f"Wrote feature table {out_df.shape} → {out_path}")


if __name__ == "__main__":
    build_feature_table()
