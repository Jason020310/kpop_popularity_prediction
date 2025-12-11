import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the dataset
data_path = './dataset/final_kpop_dataset.csv'
data = pd.read_csv(data_path)

# Define features and target
features = [
    col for col in data.columns 
    if col not in ['avg_rank', 'Artist', 'Artist_Id', 'Track_Title', 'Track_Id', 'weeks_on_chart']
]
target = 'avg_rank'

X = data[features]
y = data[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Simple Linear Regression model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Make predictions
y_pred = lr.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Visualize predictions vs actual values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, color='red')
plt.xlabel('Actual avg_rank')
plt.ylabel('Predicted avg_rank')
plt.title('Simple Linear Regression: Actual vs Predicted avg_rank')
plt.tight_layout()
plt.savefig('simple_linear_regression_avgrank.png')
plt.show()