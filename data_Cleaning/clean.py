import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import stats

# File locations – hardcoded for now, tweak later if needed
prices_csv = "prices-split-adjusted.csv"
fundamentals_csv = "fundamentals.csv"
esg_csv = "SP 500 ESG Risk Ratings.csv"
out_folder = "cleaned"

# just make the folder if it's not there already
os.makedirs(out_folder, exist_ok=True)

# load the data we care about
def grab_data():
    prices = pd.read_csv(prices_csv, parse_dates=['date'])
    funds = pd.read_csv(fundamentals_csv, parse_dates=['Period Ending'])
    esg = pd.read_csv(esg_csv)

    # drop the useless esg columns, keep the good stuff
    esg = esg[[
        'Symbol', 'Sector', 'Industry', 'Total ESG Risk score',
        'Environment Risk Score', 'Social Risk Score',
        'Governance Risk Score', 'Controversy Score'
    ]]

    # trim down fundamentals to stuff we might actually use
    funds = funds[[
        'Ticker Symbol', 'Period Ending', 'Earnings Per Share', 'Estimated Shares Outstanding',
        'Net Income', 'Total Revenue', 'Total Assets', 'Total Liabilities', 'Current Ratio', 
        'Quick Ratio', 'Operating Margin', 'Gross Margin', 'Research and Development',
        'Cash and Cash Equivalents', 'Capital Expenditures', 'Common Stocks',
        'Retained Earnings', 'Treasury Stock', 'Total Equity', 'Short-Term Investments',
        'Long-Term Debt', 'Interest Expense', 'Net Cash Flow', 'Net Cash Flow-Operating'
    ]]

    return prices, funds, esg

# basic sorting for consistency
def sort_things(p, f, e):
    p = p.sort_values(['symbol', 'date']).reset_index(drop=True)
    f = f.sort_values(['Ticker Symbol', 'Period Ending']).reset_index(drop=True)
    e = e.sort_values('Symbol').reset_index(drop=True)
    return p, f, e

# patch up missing stuff – median for numbers, mode for categories
def fill_in_blanks(df):
    num_cols = df.select_dtypes(include=[np.number]).columns
    text_cols = df.select_dtypes(include=['object', 'category']).columns

    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in text_cols:
        df[col] = df[col].fillna(df[col].mode().iloc[0])
    return df

# drop the weird outlier rows using Z-score
def ditch_outliers(df, zlimit=3.5):
    num_cols = df.select_dtypes(include=[np.number]).columns
    z = np.abs(stats.zscore(df[num_cols], nan_policy='omit'))
    keep = (z < zlimit).all(axis=1)
    return df[keep].reset_index(drop=True)

# normalize everything numeric to zero mean and unit variance
def normalize_nums(df):
    num_cols = df.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df

# the main cleaning workflow
def clean_stuff():
    # grab and sort
    price_data, fund_data, esg_data = grab_data()
    price_data, fund_data, esg_data = sort_things(price_data, fund_data, esg_data)

    # fill blanks
    price_data = fill_in_blanks(price_data)
    fund_data = fill_in_blanks(fund_data)
    esg_data = fill_in_blanks(esg_data)

    # remove garbage
    price_data = ditch_outliers(price_data)
    fund_data = ditch_outliers(fund_data)
    esg_data = ditch_outliers(esg_data)

    # scale things so ML doesn’t panic
    price_data = normalize_nums(price_data)
    fund_data = normalize_nums(fund_data)
    esg_data = normalize_nums(esg_data)

    # write everything back out
    price_data.to_csv(f"{out_folder}/prices_cleaned.csv", index=False)
    fund_data.to_csv(f"{out_folder}/fundamentals_cleaned.csv", index=False)
    esg_data.to_csv(f"{out_folder}/esgRisk_cleaned.csv", index=False)

    print("Done. Cleaned and saved everything to 'cleaned/' folder. You’re welcome.")

# run it
if __name__ == "__main__":
    clean_stuff()
