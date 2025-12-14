# KPOP Popularity Prediction
Final project for 114-1 Introduction to Data Science.   

## Abstract
In recent years, KPOP has become a significant part of the global music industry. We aimed to investigate what factors contribute to a KPOP song's popularity. To this end, we used Spotify audio features along with company and gender information to predict whether a song would be a hit. We trained both regressors (to predict a song’s ranking) and classifiers (to predict whether a song is in the top 35) using four different models and compared their performance. Furthermore, we analyzed the hit rate of songs released by different companies, as the company feature was found to be more important than audio features in some models.


## dataset sources
(Spotify Web API will no longer be able to access Audio Feature since Nov 27, 2024, so we turned to the datasets on Kaggle.)
* single_album_track_data.csv: https://www.kaggle.com/datasets/ericwan1/kpop-artists-and-full-spotify-discography
* kpop_rankings.csv: https://www.kaggle.com/datasets/romainfonta2/kpop-song-rankings/data
* kpopgroups.csv: https://www.kaggle.com/datasets/nicolsalayoarias/kpop-groups-dataset/data

## Folder Structure Overview
```
└── /kpop_popularity_prediction
    ├── /dataset
    ├── /regressor
    │   ├── regressor.py 
    │   ├── regression_actual_vs_pred.png 
    │   └── feature_importance.png
    ├── /classifier
    │   ├── classifier.py
    │   ├── classifier_comparison.png
    │   └── feature_importance.png
    └── /company_feature
        ├── prob_of_hit_by_company.py
        └──company_tier.py
```


## How to Install the Environment
1. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```
2. Activate the virtual environment:
   - On Windows:
     ```bash
     .venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run the Program
1. Ensure the virtual environment is activated.
2.  We used Linear Regression, XGBoost, LightGBM, and Random Forest to train a regressor for `avg_rank`.

    To see MSE, MAE, and R2 for each model:
    ```bash
    cd regressor
    python regressor.py
    ```
    The images illustrating the performances are also stored in `/regerssor` folder.
3.  We used Logistic Regression, XGBoost, LightGBM, and Random Forest to train a classifier for `in top35`. or not.

    To run it:
    ```bash
    cd classifier
    python classifier.py
    ```
    The images illustrating the performances are also stored in `/classifier` folder, including ROC curve and matrix confusion (they are shown in `classifier_comparison.png`)
   
## How to Reproduce the Results
1. Ensure the files are in the correct directory structure as Folder Structure Overview
2. Follow the steps in "How to Install the Environment" to set up the environment.
3. Run the program as described in "How to Run the Program."
4. The results, including feature importance and model evaluation metrics, will be displayed in the terminal and saved as visualizations (e.g., `classifier_comparison.png`).