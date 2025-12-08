# K-pop Popularity Prediction | K-pop 流行度預測

A data science project for predicting K-pop song popularity using machine learning techniques.

資料科學課程期末專案，使用機器學習技術預測 K-pop 歌曲流行度。

## 📋 Project Overview | 專案概述

This project analyzes K-pop songs using Spotify audio features and historical chart rankings to predict:
- **Average Chart Ranking** (avg_rank): How well a song ranks on charts
- **Weeks on Chart** (weeks_on_chart): How long a song stays popular

本專案分析 K-pop 歌曲，使用 Spotify 音頻特徵和歷史排行榜數據來預測：
- **平均排行榜排名** (avg_rank)：歌曲在排行榜上的表現
- **在榜週數** (weeks_on_chart)：歌曲保持流行的時間

## 🚀 Installation | 安裝

### Prerequisites | 前置需求
- Python 3.7 or higher | Python 3.7 或更高版本
- pip (Python package manager) | pip（Python 套件管理器）

### Install Dependencies | 安裝相依套件

```bash
pip install -r requirements.txt
```

## 📊 Dataset Sources | 資料集來源

* `single_album_track_data.csv`: https://www.kaggle.com/datasets/ericwan1/kpop-artists-and-full-spotify-discography
* `kpop_rankings.csv`: https://www.kaggle.com/datasets/romainfonta2/kpop-song-rankings/data
* `kpopgroups.csv`: https://www.kaggle.com/datasets/nicolsalayoarias/kpop-groups-dataset/data

### Dataset Features | 資料集特徵

The final dataset (`final_kpop_dataset.csv`) includes:
- **Audio Features**: danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration_ms, time_signature
- **Chart Performance**: avg_rank, weeks_on_chart
- **Metadata**: Artist, Artist_Id, Track_Title, Track_Id

最終資料集 (`final_kpop_dataset.csv`) 包含：
- **音頻特徵**：可舞性、能量、調性、響度、模式、語速、原聲性、器樂性、現場感、正面情緒、速度、時長、拍號
- **排行榜表現**：平均排名、在榜週數
- **元數據**：藝人、藝人ID、歌曲名、歌曲ID

## 💻 Usage | 使用方式

### 1. Simple Linear Regression Analysis | 簡單線性迴歸分析

Predicts average ranking using linear regression:

使用線性迴歸預測平均排名：

```bash
python simple_linear_regression_avgrank.py
```

**Output | 輸出:**
- Mean Squared Error printed to console | 均方誤差輸出到控制台
- Visualization saved as `simple_linear_regression_avgrank.png` | 視覺化圖表儲存為 `simple_linear_regression_avgrank.png`

### 2. Feature Importance Analysis | 特徵重要性分析

#### For Average Ranking | 針對平均排名

```bash
cd feature_importance_analysis
python feature_analysis_avgrank.py
```

**Output | 輸出:**
- Feature importance scores | 特徵重要性分數
- Visualization saved as `feature_importance_avgrank.png` | 視覺化圖表儲存為 `feature_importance_avgrank.png`

#### For Weeks on Chart | 針對在榜週數

```bash
cd feature_importance_analysis
python feature_analysis_weeksonchart.py
```

**Output | 輸出:**
- Feature importance scores | 特徵重要性分數
- Visualization saved as `feature_importance_weeks_on_chart.png` | 視覺化圖表儲存為 `feature_importance_weeks_on_chart.png`

### 3. Calculate English Percentage in Lyrics | 計算歌詞中英文比例

Calculate the percentage of English words in song lyrics:

計算歌詞中英文單字的比例：

```bash
python calculate_english_percentage.py
```

**Interactive Usage | 互動式使用:**
1. Enter lyrics line by line | 逐行輸入歌詞
2. Type `#` to finish input | 輸入 `#` 結束輸入
3. The script will display the English word percentage | 腳本將顯示英文單字佔比

**Example | 範例:**
```
請輸入歌詞，每行以 Enter 分隔，輸入 '#' 結束：
I love you so much
사랑해요
#
英文單字佔比: 83.33%
```

## 📈 Model Performance | 模型表現

The project uses two main approaches:
- **Linear Regression**: Simple baseline model for quick predictions
- **Random Forest Regressor**: Advanced model for feature importance analysis and better accuracy

本專案使用兩種主要方法：
- **線性迴歸**：快速預測的簡單基準模型
- **隨機森林迴歸器**：用於特徵重要性分析和更高準確度的進階模型

## 🔍 What Each Script Does | 各腳本功能說明

| Script | Purpose | 用途 |
|--------|---------|------|
| `simple_linear_regression_avgrank.py` | Trains a linear regression model to predict average ranking | 訓練線性迴歸模型預測平均排名 |
| `feature_analysis_avgrank.py` | Analyzes which features most influence average ranking | 分析哪些特徵對平均排名影響最大 |
| `feature_analysis_weeksonchart.py` | Analyzes which features most influence chart longevity | 分析哪些特徵對在榜時間影響最大 |
| `calculate_english_percentage.py` | Utility to calculate English word percentage in lyrics | 計算歌詞中英文單字比例的工具 |

## 📁 Project Structure | 專案結構

```
kpop_popularity_prediction/
├── dataset/                          # Data files | 資料檔案
│   ├── final_kpop_dataset.csv       # Main dataset | 主要資料集
│   ├── final_kpop_dataset_withEng.csv
│   ├── kpop_rankings.csv
│   ├── kpopgroups.csv
│   └── single_album_track_data.csv
├── feature_importance_analysis/      # Feature analysis scripts | 特徵分析腳本
│   ├── feature_analysis_avgrank.py
│   └── feature_analysis_weeksonchart.py
├── simple_linear_regression_avgrank.py
├── calculate_english_percentage.py
└── requirements.txt                  # Python dependencies | Python 相依套件
```

## 🤝 Contributing | 貢獻

This is a final project for a data science class. Feel free to fork and experiment!

這是資料科學課程的期末專案。歡迎 fork 並進行實驗！

## 📝 License | 授權

This project is for educational purposes.

本專案僅供教育用途。