import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the dataset
data_path = '../dataset/combined_kpop_dataset.csv'
try:
    data = pd.read_csv(data_path)
except FileNotFoundError:
    print("Dataset not found.")
    exit()

feature_names = [
    'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
    'duration_ms', 'time_signature', 'eng_line_ratio', 'eng_word_ratio',
    'company', 'gender' 
]
target = 'avg_rank'

data = data.fillna(0) 

X = data[feature_names]
y = data[target]

X = pd.get_dummies(X, columns=['company', 'gender'], drop_first=True)

print(f"Original number of features: {len(feature_names)}")
print(f"Number of features after encoding: {X.shape[1]}")
print("Sample of new columns:", list(X.columns[-5:])) # Look at the last few newly generated columns

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predict
y_pred = lr.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Visualize
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, color='red')
plt.xlabel('Actual avg_rank')
plt.ylabel('Predicted avg_rank')
plt.title('Simple Linear Regression: Actual vs Predicted avg_rank')
plt.tight_layout()
plt.savefig('simple_linear_regression_avgrank.png') 
plt.show()