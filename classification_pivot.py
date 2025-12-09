import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report

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

df_imputed = df_numeric.copy()

# Force conversion to numeric
for col in df_imputed.columns:
    df_imputed[col] = pd.to_numeric(df_imputed[col], errors='coerce')

# Fill missing values with mean
for col in df_imputed.columns:
    if df_imputed[col].isnull().sum() > 0:
        df_imputed[col] = df_imputed[col].fillna(df_imputed[col].mean())

# 3. Create Classification Target
# Definition of HIT: Rank <= 50 (Top 50)
target_col = 'avg_rank'
df_imputed['is_hit'] = (df_imputed[target_col] <= 50).astype(int)

# Check Class Balance
print("\n--- Class Balance ---")
print(df_imputed['is_hit'].value_counts(normalize=True))

# Prepare Features (Drop original rank and new target)
features = [col for col in df_imputed.columns if col not in [target_col, 'is_hit']]
X = df_imputed[features]
y = df_imputed['is_hit']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) 
# stratify ensures train/test have same % of hits

print(f"Training on {len(X_train)} samples")

# 4. Train Models

# --- Random Forest Classifier ---
print("\n--- Random Forest Classifier ---")
rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]

acc_rf = accuracy_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)
roc_rf = roc_auc_score(y_test, y_prob_rf)

print(f"Accuracy: {acc_rf:.4f}")
print(f"F1 Score: {f1_rf:.4f}")
print(f"ROC AUC: {roc_rf:.4f}")
print(classification_report(y_test, y_pred_rf))

# --- XGBoost Classifier ---
print("\n--- XGBoost Classifier ---")
xgb = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42, n_jobs=-1, eval_metric='logloss')
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
y_prob_xgb = xgb.predict_proba(X_test)[:, 1]

acc_xgb = accuracy_score(y_test, y_pred_xgb)
f1_xgb = f1_score(y_test, y_pred_xgb)
roc_xgb = roc_auc_score(y_test, y_prob_xgb)

print(f"Accuracy: {acc_xgb:.4f}")
print(f"F1 Score: {f1_xgb:.4f}")
print(f"ROC AUC: {roc_xgb:.4f}")


# 5. Visualization (Confusion Matrix for Best Model)
# Assuming RF is safer, but let's check who won
best_model_name = "Random Forest" if acc_rf > acc_xgb else "XGBoost"
best_pred = y_pred_rf if acc_rf > acc_xgb else y_pred_xgb

cm = confusion_matrix(y_test, best_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Hit', 'Hit'], yticklabels=['Not Hit', 'Hit'])
plt.title(f'Confusion Matrix ({best_model_name})')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('classification_confusion_matrix.png')
print(f"\nSaved classification_confusion_matrix.png ({best_model_name} won)")
