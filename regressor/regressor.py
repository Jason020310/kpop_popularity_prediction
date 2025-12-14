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
data_path = './dataset/dataset_with_feature/final_kpop_dataset.csv'
print(f"Loading data from {data_path}...")
try:
    data = pd.read_csv(data_path, encoding='utf-8')
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

# Actual vs Predicted
for name, model in models.items():
    y_pred = model.predict(X_test)
    
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)  # y=x 參考線
    plt.xlabel("Actual avg_rank")
    plt.ylabel("Predicted avg_rank")
    plt.title(f"Actual vs Predicted - {name}")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{name.replace(' ', '_')}_actual_vs_pred.png")
    plt.show()


# Feature Importance Visualization
importances_models = ["Random Forest", "XGBoost", "LightGBM"]
for name in importances_models:
    model = models[name]
    
    # RandomForest / LightGBM / XGBoost 
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feature_names = X.columns
        
        df_importance = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)
        
        plt.figure(figsize=(8, 6))
        sns.barplot(x="Importance", y="Feature", data=df_importance, palette="viridis")
        plt.title(f"Feature Importance - {name}")
        plt.tight_layout()
        plt.savefig(f"{name.replace(' ', '_')}_feature_importance.png")
        plt.show()
