import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Data
data_path = '../dataset/combined_kpop_dataset.csv'
try:
    data = pd.read_csv(data_path)
except:
    print("Error loading data.")
    exit(1)

threshold = 5
company_counts = data['company'].value_counts()
small_companies = company_counts[company_counts < threshold].index
data.loc[data['company'].isin(small_companies), 'company'] = 'Other'

audio_cols = [
    'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
    'duration_ms', 'time_signature', 'eng_line_ratio', 'eng_word_ratio'
]

threshold = 5 
company_counts = data['company'].value_counts()
small_companies = company_counts[company_counts < threshold].index
data.loc[data['company'].isin(small_companies), 'company'] = 'Other'

# Define Hit (Top 35)
data['is_hit'] = (data['avg_rank'] <= 35).astype(int)

# Calculate Hit Rate
company_stats = data.groupby('company')['is_hit'].agg(['count', 'mean']).reset_index()
company_stats.columns = ['Company', 'Song_Count', 'Hit_Rate']
company_stats = company_stats.sort_values(by='Hit_Rate', ascending=False)

# 2. Define Tiers
def get_tier(rate):
    if rate >= 0.5: return 'S-Tier (>50%)'
    elif rate >= 0.3: return 'A-Tier (30-50%)'
    else: return 'B-Tier (<30%)'

company_stats['Tier'] = company_stats['Hit_Rate'].apply(get_tier)

# 3. Plot: Tiered Visualization
plt.figure(figsize=(14, 8))
sns.scatterplot(
    data=company_stats, 
    x='Company', 
    y='Hit_Rate', 
    hue='Tier', 
    size='Song_Count', 
    sizes=(50, 500), # Circle size represents the number of songs
    palette={'S-Tier (>50%)': 'red', 'A-Tier (30-50%)': 'orange', 'B-Tier (<30%)': 'blue'}
)

# Add a horizontal average line
avg_hit_rate = data['is_hit'].mean()
plt.axhline(y=avg_hit_rate, color='grey', linestyle='--', label=f'Market Average ({avg_hit_rate:.2f})')

plt.title('K-Pop Company Tier List: Probability of Reaching Top 35', fontsize=16)
plt.ylabel('Hit Rate (Probability)', fontsize=12)
plt.xlabel('Company', fontsize=12)
plt.xticks(rotation=90)
plt.legend(title='Tier & Volume', loc='upper right')
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Annotate key companies
for i in range(len(company_stats)):
    row = company_stats.iloc[i]
    if row['Company'] in ['SM', 'YG', 'JYP', 'P NATION', 'Other']:
        plt.text(i, row['Hit_Rate']+0.02, row['Company'], ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('company_tier_list.png')