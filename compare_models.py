import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Load Data
data_path = './dataset/final_kpop_dataset_withEng.csv'
print(f"Loading data from {data_path}...")
try:
    data = pd.read_csv(data_path, encoding='latin1')
except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

# 2. Preprocessing
drop_cols = ['Artist', 'Artist_Id', 'Track_Title', 'Track_Id', 'weeks_on_chart']
existing_drop_cols = [col for col in drop_cols if col in data.columns]
df_numeric = data.drop(columns=existing_drop_cols)

# Create a copy for imputation
df_imputed = df_numeric.copy()

# Force conversion to numeric for all columns
# This handles cases where some columns might be 'object' due to dirty data
for col in df_imputed.columns:
    df_imputed[col] = pd.to_numeric(df_imputed[col], errors='coerce') # Coerce errors to NaN


# Fill missing values with mean for each column
# valid_numeric excludes columns that might have been dropped already
for col in df_imputed.columns:
    if df_imputed[col].isnull().sum() > 0:
        print(f"Imputing {df_imputed[col].isnull().sum()} missing values in {col} with mean")
        df_imputed[col] = df_imputed[col].fillna(df_imputed[col].mean())

# Verify no more missing values
df_numeric = df_imputed
print(f"Data shape after imputation: {df_numeric.shape}")

# 3. Prepare Data
target = 'avg_rank'
features = [col for col in df_numeric.columns if col != target]

X = df_numeric[features]
y = df_numeric[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training on {len(X_train)} samples, Testing on {len(X_test)} samples.")
print(f"Features: {features}")

# 4. Define Models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42),
    "LightGBM": LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42, verbose=-1)
}

# 5. Train and Evaluate
results = []

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results.append({
        "Model": name,
        "MSE": mse,
        "MAE": mae,
        "R2": r2
    })
    print(f"{name} -> MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

# 6. Comparison Visualization
df_results = pd.DataFrame(results)

# Reshape for seaborn
df_melted = df_results.melt(id_vars="Model", var_name="Metric", value_name="Score")

plt.figure(figsize=(12, 6))
sns.barplot(x="Metric", y="Score", hue="Model", data=df_melted, palette="magma")
plt.title("Model Performance Comparison (Lower MSE/MAE is better, Higher R2 is better)")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("model_comparison.png")
print("\nSaved model_comparison.png")

# Save detailed results to CSV
df_results.to_csv("model_comparison_results.csv", index=False)
print("Saved model_comparison_results.csv")
