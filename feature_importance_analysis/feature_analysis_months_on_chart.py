import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 1. Load Data
data_path = '../dataset/combined_kpop_dataset.csv'
try:
    data = pd.read_csv(data_path)
except FileNotFoundError:
    print(f"Dataset not found: {data_path}")
    exit()

# 2. Define Features 
feature_cols = [
    'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
    'duration_ms', 'time_signature', 'eng_line_ratio', 'eng_word_ratio',
    'company', 'gender' 
]
target = 'months_on_chart'

# Filter small companies
threshold = 10 
company_counts = data['company'].value_counts()

small_companies = company_counts[company_counts < threshold].index
data.loc[data['company'].isin(small_companies), 'company'] = 'Other'

print(f"Merged {len(small_companies)} 間small companies into 'Other'.") 

X = data[feature_cols]
y = data[target]

# 3. Encoding 
X = pd.get_dummies(X, columns=['company', 'gender'], drop_first=True)

feature_names = X.columns.tolist()

# 4. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train
rf = RandomForestRegressor(random_state=42, n_jobs=-1) 
rf.fit(X_train, y_train)

# 6. Evaluate
y_pred = rf.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")

# 7. Feature Importance Analysis
feature_importances = rf.feature_importances_
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)


top_20_features = importance_df.head(20)

print("Top 10 Feature Importances:")
print(top_20_features.head(10))

# Visualize
plt.figure(figsize=(10, 8)) 
plt.barh(top_20_features['Feature'], top_20_features['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Top 20 Feature Importance for months_on_chart')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance_months_on_chart.png')
plt.show()