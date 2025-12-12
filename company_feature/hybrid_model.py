import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

'''
結合了「公司/性別特徵」與「音樂特徵」來跑 XGBoost，預測目標是 Top 35
'''

# 1. Load Data
data_path = '../dataset/combined_kpop_dataset.csv'
try:
    data = pd.read_csv(data_path)
except:
    exit(1)

# 2. Preprocessing
threshold = 5
company_counts = data['company'].value_counts()
small_companies = company_counts[company_counts < threshold].index
data.loc[data['company'].isin(small_companies), 'company'] = 'Other'

audio_cols = [
    'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
    'duration_ms', 'time_signature', 'eng_line_ratio', 'eng_word_ratio'
]

data[audio_cols] = data[audio_cols].fillna(data[audio_cols].mean())

data['energy_dance'] = data['energy'] * data['danceability']
data['english_speech'] = data['eng_word_ratio'] * data['speechiness']

# (D) Set Target: Top 35
target_col = 'avg_rank'
bins = [0, 35, 9999]
labels = [0, 1]
label_names = {0: 'Top 35 (Hit)', 1: 'Below 35'}

data['rank_category'] = pd.cut(data[target_col], bins=bins, labels=labels)
data = data.dropna(subset=['rank_category'])
y = data['rank_category'].astype(int)

# (E) Prepare X (including Company + Gender + Audio Features + Combined Features)
features_to_use = ['company', 'gender'] + audio_cols + ['energy_dance', 'english_speech']
X = data[features_to_use]

# One-Hot Encoding
X_encoded = pd.get_dummies(X, columns=['company', 'gender'], drop_first=True)

# 3. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training on {len(X_train)} samples with {X_encoded.shape[1]} features")

# 4. Train XGBoost
print("\n--- Training XGBoost (Hybrid) ---")
xgb = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss'
)

xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)

# 5. Evaluation
acc = accuracy_score(y_test, y_pred)
print(f"\n=== Hybrid Model Results ===")
print(f"Accuracy: {acc:.4f}")

unique_labels = sorted(list(set(y_test) | set(y_pred)))
current_names = [label_names[i] for i in unique_labels]

print("\nClassification Report:")
print(classification_report(y_test, y_pred, labels=unique_labels, target_names=current_names, zero_division=0))

# 6. Feature Importance 
importances = xgb.feature_importances_
feature_names = X_encoded.columns
feature_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)

print("\n--- Top 15 Most Important Features ---")
print(feature_imp_df.head(15))

plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=feature_imp_df.head(15), palette='viridis')
plt.title('Feature Importance (Audio + Company)')
plt.show()