# K-pop Popularity Prediction

A machine learning project that predicts K-pop song popularity based on audio features. This tool can predict:
- **Average Chart Rank** (`avg_rank`): How high a song will rank on charts (lower is better)
- **Weeks on Chart** (`weeks_on_chart`): How long a song will stay on charts (higher is better)

Final project for data science class.

## Features

- 🎵 Train machine learning models on K-pop song data
- 📊 Multiple model comparison (Linear Regression, Random Forest, Gradient Boosting)
- 🎯 Interactive prediction tool for new songs
- 📈 Visualization of model performance
- 📝 Feature importance analysis

## Installation

### Requirements
- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Jason020310/kpop_popularity_prediction.git
cd kpop_popularity_prediction
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Train Models

First, train the machine learning models on the K-pop dataset:

```bash
python train_models.py
```

This will:
- Train multiple models (Linear Regression, Random Forest, Gradient Boosting)
- Compare their performance
- Save the best models to the `./models/` directory
- Generate prediction visualization plots

**Output:**
- `./models/avg_rank_model.pkl` - Model for predicting average chart rank
- `./models/weeks_on_chart_model.pkl` - Model for predicting chart longevity
- `./models/avg_rank_predictions.png` - Visualization of rank predictions
- `./models/weeks_on_chart_predictions.png` - Visualization of weeks predictions

### 2. Predict Song Popularity

Use the trained models to predict popularity for new songs:

```bash
python predict_popularity.py
```

The tool will guide you through entering song audio features. You can obtain these features from:
- Spotify API
- Audio analysis tools
- Music information databases

**Example session:**
```
Enter song audio features
==========================================================
danceability (0-1, how suitable for dancing): 0.75
energy (0-1, intensity and activity): 0.85
key (0-11, pitch class): 5
loudness (dB, typically -60 to 0): -4.5
mode (0=minor, 1=major): 1
speechiness (0-1, presence of spoken words): 0.06
acousticness (0-1, acoustic vs electronic): 0.12
instrumentalness (0-1, likelihood of no vocals): 0.0
liveness (0-1, presence of audience): 0.15
valence (0-1, musical positiveness): 0.68
tempo (BPM, typically 60-200): 125
duration_ms (milliseconds): 210000
time_signature (typically 3-7): 4

POPULARITY PREDICTION RESULTS
==========================================================
Predicted Average Rank: 32.5
  → Lower rank = Higher popularity (1 is best)
  Interpretation: Very Good! Strong chart performance expected ✨

Predicted Weeks on Chart: 12.3 weeks
  Interpretation: Great staying power! 🎶
```

### 3. Analyze Feature Importance

Explore which features most influence popularity:

```bash
cd feature_importance_analysis
python feature_analysis_avgrank.py      # For chart rank analysis
python feature_analysis_weeksonchart.py  # For longevity analysis
```

### 4. Additional Scripts

- **Simple Linear Regression Analysis:**
  ```bash
  python simple_linear_regression_avgrank.py
  ```

- **Calculate English Percentage in Lyrics:**
  ```bash
  python calculate_english_percentage.py
  ```

## Dataset Sources

The project uses data from multiple Kaggle datasets:
* **single_album_track_data.csv**: [K-pop Artists and Full Spotify Discography](https://www.kaggle.com/datasets/ericwan1/kpop-artists-and-full-spotify-discography)
* **kpop_rankings.csv**: [K-pop Song Rankings](https://www.kaggle.com/datasets/romainfonta2/kpop-song-rankings/data)
* **kpopgroups.csv**: [K-pop Groups Dataset](https://www.kaggle.com/datasets/nicolsalayoarias/kpop-groups-dataset/data)

The combined dataset includes:
- Audio features (danceability, energy, tempo, etc.)
- Chart performance metrics
- Artist information

## Audio Features Explained

| Feature | Description | Range |
|---------|-------------|-------|
| danceability | How suitable a track is for dancing | 0.0 - 1.0 |
| energy | Intensity and activity measure | 0.0 - 1.0 |
| key | The key the track is in (pitch class) | 0 - 11 |
| loudness | Overall loudness in decibels (dB) | -60 to 0 |
| mode | Major (1) or minor (0) modality | 0 or 1 |
| speechiness | Presence of spoken words | 0.0 - 1.0 |
| acousticness | Acoustic vs electronic | 0.0 - 1.0 |
| instrumentalness | Predicts if track has no vocals | 0.0 - 1.0 |
| liveness | Presence of audience in recording | 0.0 - 1.0 |
| valence | Musical positiveness/happiness | 0.0 - 1.0 |
| tempo | Overall tempo in BPM | ~60 - 200 |
| duration_ms | Duration in milliseconds | Variable |
| time_signature | Estimated time signature | 3 - 7 |

## Project Structure

```
kpop_popularity_prediction/
├── dataset/                          # Data files
│   ├── final_kpop_dataset.csv
│   ├── final_kpop_dataset_withEng.csv
│   └── ...
├── feature_importance_analysis/      # Feature analysis scripts
│   ├── feature_analysis_avgrank.py
│   └── feature_analysis_weeksonchart.py
├── models/                           # Trained models (generated)
│   ├── avg_rank_model.pkl
│   └── weeks_on_chart_model.pkl
├── train_models.py                   # Model training script
├── predict_popularity.py             # Prediction interface
├── simple_linear_regression_avgrank.py
├── calculate_english_percentage.py
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## Model Performance

The trained models are evaluated using multiple metrics:
- **MSE (Mean Squared Error)**: Lower is better
- **R² Score**: Closer to 1.0 is better (explains variance)
- **MAE (Mean Absolute Error)**: Average prediction error

The best performing model is automatically selected and saved for predictions.

## Contributing

This is a student project. Contributions and suggestions are welcome!

## License

This project is created for educational purposes.