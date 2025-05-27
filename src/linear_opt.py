# Optimization for linear regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, classification_report

# Load cleaned dataset
df = pd.read_csv("final_clean_dataset.csv")

# Select features and labels
features = ['open', 'high', 'low', 'close', 'volume',
            'Net Income', 'Total Revenue', 'Total Assets',
            'Gross Margin', 'Total ESG Risk score']
X = df[features].values
y = df['action'].values

# Add the column of 1 values to X
X = np.c_[np.ones((X.shape[0], 1)), X]

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Do linear regression using 32-batch-size GD
class LinearRegression32GD:
    def __init__(self):
        self.theta = None

    def fit(self, X_train, y_train, X_test, y_test, lr=0.001, epochs=100, batch_size=32):
        # randomize the initial weight parameters for later optimize
        self.theta = np.random.randn(X_train.shape[1])
        # get the size of the dataset
        m = X_train.shape[0]

        # Create empty lists for recording the MSE scores of
        # the linear regression model on training data & test data
        mse_train_history = []
        mse_test_history = []

        # Iterate over the batches in each epoch
        for epoch in range(epochs):
            # This line generates random indices from 0 to m (the total number of samples in the training set) without replacement.
            # m is the size of the training dataset. The number of indices generated is equal to the batch_size.
            indices = np.random.choice(m, batch_size)
            # creates a batch of input features for training
            X_batch = X_train[indices]
            # selects the corresponding subset of y values
            y_batch = y_train[indices]

            # generating the predictions on X_batch
            y_pred = X_batch.dot(self.theta)
            # compute the gradient of the loss function
            error = y_pred - y_batch
            gradient = 2 / batch_size * X_batch.T.dot(error)
            # updating the weights with the computed gradient
            self.theta -= lr * gradient

            # Predict
            y_train_pred = X_train.dot(self.theta)
            y_test_pred = X_test.dot(self.theta)

            # Record the MSE scores
            mse_train_history.append(mean_squared_error(y_train, y_train_pred))
            mse_test_history.append(mean_squared_error(y_test, y_test_pred))

        return mse_train_history, mse_test_history

# Train using 32-batch-size GD
model = LinearRegression32GD()
mse_train, mse_test = model.fit(X_train, y_train, X_test, y_test)

# Plot MSE over epochs
plt.figure(figsize=(10, 6))
plt.plot(mse_train, label='Train MSE')
plt.plot(mse_test, label='Test MSE')
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.title("Train and Test MSE")
plt.legend()
plt.grid(True)
plt.show()

# Evaluate classification performance
y_pred = np.dot(X_test, model.theta)

# Based on 1% Rule
def action_classify(val):
    if val > 0.01:
        return 1
    elif val < -0.01:
        return -1
    else:
        return 0

# Turn the value to class label
y_pred_class = []
for val in y_pred:
    y_pred_class.append(action_classify(val))

print("Optimized Linear Regression Model Report")
print(classification_report(y_test, y_pred_class))

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
plt.title("Optimized Linear Regression Model Report")
plt.tight_layout()
plt.savefig("Opt_LR_report.png")
plt.show()

# Plot Distribution Graph
plt.hist(y_pred, bins=100)
plt.title("Distribution of Optimized Linear Regression Predictions")
plt.xlabel("Predicted Value")
plt.ylabel("Amount")
plt.show()
