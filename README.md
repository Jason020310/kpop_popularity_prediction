# KPOP Popularity Prediction
Final project for 114-1 Introduction to Data Science

## dataset sources
* single_album_track_data.csv: https://www.kaggle.com/datasets/ericwan1/kpop-artists-and-full-spotify-discography
* kpop_rankings.csv: https://www.kaggle.com/datasets/romainfonta2/kpop-song-rankings/data
* kpopgroups.csv: https://www.kaggle.com/datasets/nicolsalayoarias/kpop-groups-dataset/data

## Folder Structure

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
2. Run the classifier script:
   ```bash
   python classifier.py
   ```

## How to Reproduce the Results
1. Ensure the dataset files are in the correct directory structure as follows:
   ```
   dataset/
       combined_kpop_dataset.csv
       original_dataset/
           kpop_rankings.csv
           kpopgroups.csv
           single_album_track_data.csv
   ```
2. Follow the steps in "How to Install the Environment" to set up the environment.
3. Run the program as described in "How to Run the Program."
4. The results, including feature importance and model evaluation metrics, will be displayed in the terminal and saved as visualizations (e.g., `classifier_comparison.png`).