import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load Data
data_path = '../dataset/combined_kpop_dataset.csv'
print(f"Loading data from {data_path}...")
try:
    data = pd.read_csv(data_path)
except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

# 2. Preprocessing
# Keep only identity features (Company + Gender) ---
feature_cols = ['company', 'gender'] 
target_col = 'avg_rank'

# --- Company Name Processing) ---
threshold = 5 
company_counts = data['company'].value_counts()
small_companies = company_counts[company_counts < threshold].index
data.loc[data['company'].isin(small_companies), 'company'] = 'Other'

print(f"Companies being analyzed: {data['company'].nunique()}")
print(data['company'].unique()) # See which companies are retained

# --- Define Hit (Top 35) ---
bins = [0, 35, 9999] 
labels = [0, 1] 
label_names = {0: 'Top 35 (Hit)', 1: 'Below 35'}

data['rank_category'] = pd.cut(data[target_col], bins=bins, labels=labels)
data = data.dropna(subset=['rank_category'])

X = data[feature_cols]
y = data['rank_category'].astype(int)

# One-Hot Encoding on Company and Gender
X_encoded = pd.get_dummies(X, columns=['company', 'gender'], drop_first=True)

# 3. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining on {len(X_train)} samples using ONLY Company & Gender")

# 4. Train Model (Random Forest)
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight='balanced')
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

# 5. Evaluation
acc = accuracy_score(y_test, y_pred)
print(f"\n=== Company-Only Model Results ===")
print(f"Accuracy: {acc:.4f}")

unique_labels = sorted(list(set(y_test) | set(y_pred)))
current_names = [label_names[i] for i in unique_labels]

print("\nClassification Report:")
print(classification_report(y_test, y_pred, labels=unique_labels, target_names=current_names, zero_division=0))

# 6. Calculate the Hit Rate for each company
print("\n--- Hit Rate Analysis by Company (Truth) ---")

# Create a temporary dataframe
analysis_df = data.copy()
analysis_df['is_hit'] = (analysis_df['rank_category'] == 0).astype(int) # 0 is Hit

# Calculate the total number of songs and hits for each company
company_stats = analysis_df.groupby('company')['is_hit'].agg(['count', 'sum', 'mean']).sort_values(by='mean', ascending=False)
company_stats.columns = ['Total Songs', 'Hit Songs', 'Hit Rate']
print(company_stats)

# Plot
plt.figure(figsize=(12, 6))
sns.barplot(x=company_stats.index, y=company_stats['Hit Rate'], palette='viridis')
plt.title('Probability of reaching Top 35 by Company')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Hit Rate (0.0 - 1.0)')
plt.tight_layout()
plt.savefig('prob_of_hit_by_company.png')