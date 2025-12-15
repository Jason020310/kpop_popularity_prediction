# KPOP Popularity Prediction
Final project for NCU 114-1 Introduction to Data Science.   

## Abstract
In recent years, KPOP has become a significant part of the global music industry. We aimed to investigate what factors contribute to a KPOP song's popularity. To this end, we used Spotify audio features along with company and gender information to predict whether a song would be a hit. We trained both regressors (to predict a song’s ranking) and classifiers (to predict whether a song is in the top 35) using four different models and compared their performance. Furthermore, we analyzed the hit rate of songs released by different companies, as the company feature was found to be more important than audio features in some models.


## Dataset Sources
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
    The images illustrating the performances are also stored in `/classifier` folder, including ROC curve and confusion matrix (they are shown in `classifier_comparison.png`)
   
## How to Reproduce the Results
1. Ensure the files are in the correct directory structure as Folder Structure Overview
2. Follow the steps in "How to Install the Environment" to set up the environment.
3. Run the program as described in "How to Run the Program."
4. The results, including feature importance and model evaluation metrics, will be displayed in the terminal and saved as visualizations (e.g., `classifier_comparison.png`).

## Results
### Regressor
| Model             |   MSE     |  MAE    |   R2    |
|------------------|-----------|:-------:|:-------:|
| Linear Regression| 1471.9802 | 31.8449 | 0.0058  |
| Random Forest    | 1479.6589 | 30.5514 | 0.0006  |
| XGBoost          | 1599.0760 | 31.5336 | -0.0801 |
| LightGBM         | 1561.3749 | 31.1130 | -0.0546 |


The predictive performance of regressors:
![images](https://github.com/Jason020310/kpop_popularity_prediction/blob/master/regressor/Linear_Regression_actual_vs_pred.png)
![images](https://github.com/Jason020310/kpop_popularity_prediction/blob/master/regressor/XGBoost_actual_vs_pred.png)
![images](https://github.com/Jason020310/kpop_popularity_prediction/blob/master/regressor/Random_Forest_actual_vs_pred.png)
![images](https://github.com/Jason020310/kpop_popularity_prediction/blob/master/regressor/LightGBM_actual_vs_pred.png)

### Classifier
| Model               | Accuracy | Precision | Recall | F1-score |
|---------------------|:--------:|:---------:|:------:|:--------:|
| Logistic Regression | 0.62     | 0.64      | 0.62   | 0.62     |
| Random Forest       | 0.59     | 0.56      | 0.59   | 0.57     |
| XGBoost             | 0.58     | 0.56      | 0.58   | 0.57     |
| LightGBM            | 0.59     | 0.57      | 0.59   | 0.58     |'

The ROC curve and confusion matrix of classifiers:
![iamges](https://github.com/Jason020310/kpop_popularity_prediction/blob/master/classifier/classifier_comparison.png)
