# ECS-171-Group18 Final Project Update: 5/30

## Comparative Analysis Notes
To ensure fair and consistent model comparison, please notice that:

- Use the same dataset version:
Since we find some unscaled data for the dataset uploaded before, you can use the ```merged.py``` to merge the unified dataset for comparative analysis (under temp_dataset direcotry).
It includes Jashan’s engineer features and all unscaled data are scaled by using ```scaler = StandardScaler()```, which can be directly used for modeling.

- Make sure to use the same set of features during testing (as following code), so results are comparable across models.

- Use a unified labeling format (Buy / Hold / Sell).
If different labeling methods are used (binary vs. three), three models will not be directly comparable. If you use the ```merged.py```, it pre-label all datas so you can ignore this step.
Labeling rule is explained in ```Labeling Clarification``` section if you want to check with.

Features we considered right now:
```python
features = ['open', 'high', 'low', 'volume',
            'Net Income', 'Total Revenue', 'Total Assets', 'Total Liabilities','Operating Margin',
            'Gross Margin','Total ESG Risk score','Environment Risk Score',
            'Social Risk Score', 'Governance Risk Score','Controversy Score','Current Ratio', 'Quick Ratio',
            'ret_1d','vol_5d','ma_5d',
            # if decide to use leaked data, use following features:
            'close', 'fiveday_close',
            ]
```
Split test and train as:
```python
train_test_split(X, y, test_size=0.2, random_state=42)
```

## Labeling Clarification

The `action` label is encoded to simplify the model training process:

- **Buy** = `1`  
- **Hold** = `0`  
- **Sell** = `-1`

Classification is based on the **1% Rule in Trading**:

- If `pct_change > 1%` → **Buy**  
- If `pct_change < -1%` → **Sell**  
- Otherwise → **Hold**

### Formula for `pct_change`:

```python
pct_change = (fiveday_close - close) / close
```

Where:
- `close` = 1st-day closing price  
- `fiveday_close` = 5th-day closing price

Which accomplish a short-term stock action prediction over a 5-day period.

