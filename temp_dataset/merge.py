import pandas as pd
import numpy as np
# Load datasets
fundamentals_df = pd.read_csv("fundamentals_cleaned_scaled_v2.csv")
prices_df = pd.read_csv("prices_cleaned_scaled.csv")
esgRisk_df = pd.read_csv("esgRisk_cleaned_scaled_v2.csv")

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators: lagged returns, moving average, rolling std dev."""
    df['ret_1d'] = df.groupby('symbol')['close'].pct_change(1)
    df['ma_5d'] = df.groupby('symbol')['close'].transform(lambda x: x.rolling(5).mean())
    df['vol_5d'] = df.groupby('symbol')['close'].transform(lambda x: x.rolling(5).std())
    return df

# Prepare fundamentals
fundamentals_df['Period Ending'] = pd.to_datetime(fundamentals_df['Period Ending'])
fundamentals_df['year'] = fundamentals_df['Period Ending'].dt.year

# Prepare prices
prices_df['date'] = pd.to_datetime(prices_df['date'])
prices_df.sort_values(['symbol', 'date'], inplace=True)

# add engineer features
prices_df = engineer_features(prices_df)

prices_df['fiveday_close'] = prices_df.groupby('symbol')['close'].shift(-5)
prices_filtered = prices_df.dropna(subset=['fiveday_close']).copy()
prices_filtered['year'] = prices_filtered['date'].dt.year
prices_filtered.rename(columns={'symbol': 'Ticker Symbol'}, inplace=True)

# Merge prices with fundamentals
merged_df = pd.merge(prices_filtered, fundamentals_df, on=['Ticker Symbol', 'year'], how='left')

# Merge with ESG
esg_df = esgRisk_df.rename(columns={'Symbol': 'Ticker Symbol'})
final_df = pd.merge(merged_df, esg_df, on='Ticker Symbol', how='left')

# Calculate pct_change and new action label with 1% threshold
final_df['pct_change'] = (final_df['fiveday_close'] - final_df['close']) / final_df['close']

def action_classify(change):
    if change > 0.01:
        return 'Buy'
    elif change < -0.01:
        return 'Sell'
    else:
        return 'Hold'

final_df['action'] = final_df['pct_change'].apply(action_classify)

# Encode (Buy = 1, Hold = 0, Sell = -1)
action_mapping = {'Sell': -1, 'Hold': 0, 'Buy': 1}
final_df['action'] = final_df['action'].map(action_mapping)

# # Export full training dataset
# features = [
#     'Ticker Symbol', 'date', 'close', 'fiveday_close', 'pct_change', 'action',
#     'open', 'high', 'low', 'volume',
#     'Net Income', 'Total Revenue', 'Total Assets',
#     'Gross Margin', 'Total ESG Risk score', 'Environment Risk Score',
#     'Social Risk Score', 'Governance Risk Score',
#     'Controversy Score'
# ]
# features = [col for col in features if col in final_df.columns]


# Abstract numerical features
features = final_df.select_dtypes(include=[np.number]).columns.tolist()
features.remove('action')

# Delete empty lines
clean_df = final_df.dropna(subset=features + ['action'])

# Export full training dataset
clean_df.to_csv("full_dataset_1pct.csv", index=False)


print("saved successfully.")

