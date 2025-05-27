import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("final_clean_dataset.csv")

# Select features and labels
features = ['open', 'high', 'low', 'close', 'volume',
            'Net Income', 'Total Revenue', 'Total Assets',
            'Gross Margin', 'Total ESG Risk score']

X = df[features]
# (Buy = 1, Hold = 0, Sell = -1)
y = df['action']

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train linear regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Do the prediction
y_pred = linear_model.predict(X_test)

# Based on 1% Rule
def action_classify(val):
    if val > 0.01:
        return 1   # Buy
    elif val < -0.01:
        return -1  # Sell
    else:
        return 0   # Hold

# Turn the value to class label
y_pred_class = []
for val in y_pred:
    y_pred_class.append(action_classify(val))


# print classification_report
print("Linear Regression Model Report")
print(classification_report(y_test, y_pred_class, target_names=['Sell (-1)', 'Hold (0)', 'Buy (1)']))

# Turn to dictionary format so that can use DataFrame
report_dict = classification_report(y_test, y_pred_class, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
# Rename encoded value to actual action labels
index_map = {
    "-1": "Sell (-1)",
    "0": "Hold (0)",
    "1": "Buy (1)"
}
report_df.rename(index=index_map, inplace=True)

# Plot
plt.figure(figsize=(8, 6))
plt.table(cellText=report_df.round(2).values,
          rowLabels=report_df.index,
          colLabels=report_df.columns,
          cellLoc='center',
          loc='center')
plt.axis('off')
plt.title("Linear Regression Model Report")
plt.tight_layout()
plt.savefig("LR_report.png")
plt.show()

# Plot Distribution Graph
plt.hist(y_pred, bins=100)
plt.title("Distribution of Linear Regression Predictions")
plt.xlabel("Predicted Value")
plt.ylabel("Amount")
plt.show()
