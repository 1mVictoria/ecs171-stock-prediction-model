# ECS-171-Group18 Final Project

Features we considered right now:
```python
features = ['open', 'high', 'low', 'close', 'volume',
            'Net Income', 'Total Revenue', 'Total Assets',
            'Gross Margin', 'Total ESG Risk score']

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

