import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error

# 1. Load Data
data_path = './dataset/final_kpop_dataset_withEng.csv'
print(f"Loading data from {data_path}...")
try:
    data = pd.read_csv(data_path, encoding='latin1')
except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

# 2. Preprocessing & Imputation
drop_cols = ['Artist', 'Artist_Id', 'Track_Title', 'Track_Id', 'weeks_on_chart']
existing_drop_cols = [col for col in drop_cols if col in data.columns]
df_numeric = data.drop(columns=existing_drop_cols)

# Create a copy for imputation
df_imputed = df_numeric.copy()

# Force conversion to numeric
for col in df_imputed.columns:
    df_imputed[col] = pd.to_numeric(df_imputed[col], errors='coerce')

# Fill missing values with mean
for col in df_imputed.columns:
    if df_imputed[col].isnull().sum() > 0:
        df_imputed[col] = df_imputed[col].fillna(df_imputed[col].mean())

# 3. Train/Test Split
target = 'avg_rank'
features = [col for col in df_imputed.columns if col != target]

X = df_imputed[features]
y = df_imputed[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training on {len(X_train)} samples")

# 4. Hyperparameter Tuning using RandomizedSearchCV

# --- Random Forest Tuning ---
print("\n--- Tuning Random Forest ---")
rf_param_dist = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [None, 10, 20, 30],         # Control depth to prevent overfitting
    'min_samples_split': [2, 5, 10],         # Require more samples to split
    'min_samples_leaf': [1, 2, 4],           # Require more samples in leaves (smooths model)
    'max_features': ['sqrt', 'log2', None]   # Random subset of features
}

rf = RandomForestRegressor(random_state=42, n_jobs=-1)
rf_search = RandomizedSearchCV(estimator=rf, param_distributions=rf_param_dist, 
                               n_iter=20, cv=5, verbose=1, random_state=42, n_jobs=-1, scoring='r2')

rf_search.fit(X_train, y_train)

best_rf = rf_search.best_estimator_
y_pred_rf = best_rf.predict(X_test)
rf_r2 = r2_score(y_test, y_pred_rf)

print(f"Best RF Parameters: {rf_search.best_params_}")
print(f"Best RF R2 (Test Set): {rf_r2:.4f}")


# --- XGBoost Tuning ---
print("\n--- Tuning XGBoost ---")
xgb_param_dist = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5, 6, 8],
    'subsample': [0.6, 0.8, 1.0],         # Prevent overfitting
    'colsample_bytree': [0.6, 0.8, 1.0],  # Prevent overfitting
    'gamma': [0, 0.1, 0.2],               # Regularization (Minimum loss reduction)
    'reg_lambda': [1, 1.5, 2]             # L2 Regularization
}

xgb = XGBRegressor(random_state=42, n_jobs=-1)
xgb_search = RandomizedSearchCV(estimator=xgb, param_distributions=xgb_param_dist,
                                n_iter=20, cv=5, verbose=1, random_state=42, n_jobs=-1, scoring='r2')

xgb_search.fit(X_train, y_train)

best_xgb = xgb_search.best_estimator_
y_pred_xgb = best_xgb.predict(X_test)
xgb_r2 = r2_score(y_test, y_pred_xgb)

print(f"Best XGB Parameters: {xgb_search.best_params_}")
print(f"Best XGB R2 (Test Set): {xgb_r2:.4f}")

# Return comparison
print("\n--- Tuning Results ---")
print(f"Random Forest (Tuned): {rf_r2:.4f}")
print(f"XGBoost (Tuned): {xgb_r2:.4f}")
