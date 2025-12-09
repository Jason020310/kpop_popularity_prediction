import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.metrics import r2_score

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

# Force conversion to numeric for all columns
for col in df_imputed.columns:
    df_imputed[col] = pd.to_numeric(df_imputed[col], errors='coerce') 

# Fill missing values with mean
for col in df_imputed.columns:
    if df_imputed[col].isnull().sum() > 0:
        print(f"Imputing {df_imputed[col].isnull().sum()} missing values in {col} with mean")
        df_imputed[col] = df_imputed[col].fillna(df_imputed[col].mean())

# 3. train/test split
target = 'avg_rank'
features = [col for col in df_imputed.columns if col != target]

X = df_imputed[features]
y = df_imputed[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training on {len(X_train)} samples")

# 4. Random Forest Feature Importance
print("\n--- Training Random Forest for Feature Importance ---")
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# Initial Performance
y_pred = rf.predict(X_test)
initial_r2 = r2_score(y_test, y_pred)
print(f"Full Model R2: {initial_r2:.4f}")

# Extract Importances
importances = rf.feature_importances_
feature_imp_df = pd.DataFrame({'Feature': features, 'Importance': importances})
feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=False)

print("\n--- Feature Importance (Top 10) ---")
print(feature_imp_df.head(10))

plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=feature_imp_df, palette='viridis')
plt.title(f'Random Forest Feature Importance (R2: {initial_r2:.4f})')
plt.tight_layout()
plt.savefig('imputed_feature_importance.png')
print("Saved imputed_feature_importance.png")

# 5. Recursive Feature Elimination (RFE)
print("\n--- Recursive Feature Elimination (Top 5 Features) ---")
rfe = RFE(estimator=RandomForestRegressor(n_estimators=50, random_state=42), n_features_to_select=5)
rfe.fit(X_train, y_train)

selected_features = [f for f, s in zip(features, rfe.support_) if s]
print(f"Top 5 Features selected by RFE: {selected_features}")

# Validating Top 5
rf_top5 = RandomForestRegressor(n_estimators=100, random_state=42)
rf_top5.fit(X_train[selected_features], y_train)
y_pred_top5 = rf_top5.predict(X_test[selected_features])
r2_top5 = r2_score(y_test, y_pred_top5)
print(f"Top 5 Features Model R2: {r2_top5:.4f}")
