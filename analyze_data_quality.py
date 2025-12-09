import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load Data
data_path = './dataset/final_kpop_dataset_withEng.csv'
print(f"Loading data from {data_path}...")
try:
    # Use latin1 as identified previously
    raw_data = pd.read_csv(data_path, encoding='latin1')
except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

print(f"\n--- 1. Data Shape ---")
print(f"Raw Rows: {len(raw_data)}")
print(f"Columns: {list(raw_data.columns)}")

# 2. Missing Value Analysis
print(f"\n--- 2. Missing Values ---")
missing = raw_data.isnull().sum()
print(missing[missing > 0])

# Check how many rows remain after dropping NaNs
clean_data = raw_data.dropna()
print(f"\nRows after dropna(): {len(clean_data)}")
print(f"Data Loss: {100 * (1 - len(clean_data)/len(raw_data)):.2f}%")

# 3. Correlation Analysis
# Filter numeric columns
numeric_data = clean_data.select_dtypes(include=[np.number])
# Exclude identifiers if they accidentally got parsed as numeric (e.g. IDs) or target
if 'weeks_on_chart' in numeric_data.columns:
    numeric_data = numeric_data.drop(columns=['weeks_on_chart']) # drop potential leak/target

target = 'avg_rank'
if target in numeric_data.columns:
    print(f"\n--- 3. Correlation with {target} (Top 10) ---")
    correlations = numeric_data.corr()[target].abs().sort_values(ascending=False)
    print(correlations.head(10))
    
    # Save correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix (Cleaned Data)')
    plt.tight_layout()
    plt.savefig('data_quality_correlation.png')
    print("Saved data_quality_correlation.png")
else:
    print(f"Target '{target}' not found in numeric columns!")

# 4. Target Distribution
if target in clean_data.columns:
    plt.figure(figsize=(8, 5))
    sns.histplot(clean_data[target], bins=20, kde=True)
    plt.title(f'Distribution of {target}')
    plt.xlabel('Average Rank')
    plt.savefig('target_distribution.png')
    print("Saved target_distribution.png")

# 5. Check "English_percent" specific issues
if 'English_percent' in raw_data.columns:
    print("\n--- 5. English_percent Status ---")
    print(raw_data['English_percent'].describe())
    # Check if it's actually numeric or string
    print(f"Dtype: {raw_data['English_percent'].dtype}")
