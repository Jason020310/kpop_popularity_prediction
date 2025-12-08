"""
K-pop Popularity Prediction Tool

This script allows users to predict the popularity of K-pop songs based on their audio features.
It uses pre-trained machine learning models to predict:
- avg_rank: Average ranking position on charts (lower is better)
- weeks_on_chart: Number of weeks a song stays on charts (higher is better)

Usage:
    python predict_popularity.py
    
The script will prompt you to enter song features interactively.
"""

import pickle
import os
import sys
import pandas as pd
import numpy as np


def load_model(target_name):
    """Load a trained model from disk."""
    model_path = f'./models/{target_name}_model.pkl'
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please run 'python train_models.py' first to train the models.")
        sys.exit(1)
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    return model_data


def get_feature_input(feature_name, feature_info):
    """Get user input for a specific feature with validation."""
    while True:
        try:
            prompt = f"{feature_name}"
            if feature_info:
                prompt += f" {feature_info}"
            prompt += ": "
            
            value = input(prompt).strip()
            
            # Convert to float
            value = float(value)
            
            # Basic validation
            if feature_name in ['danceability', 'energy', 'speechiness', 'acousticness', 
                               'instrumentalness', 'liveness', 'valence']:
                if not 0 <= value <= 1:
                    print(f"  Warning: {feature_name} should be between 0 and 1")
                    continue
            elif feature_name == 'key':
                if not 0 <= value <= 11:
                    print(f"  Warning: key should be between 0 and 11")
                    continue
            elif feature_name == 'mode':
                if value not in [0, 1]:
                    print(f"  Warning: mode should be 0 (minor) or 1 (major)")
                    continue
            elif feature_name == 'time_signature':
                if not 3 <= value <= 7:
                    print(f"  Warning: time_signature typically ranges from 3 to 7")
            
            return value
            
        except ValueError:
            print("  Invalid input. Please enter a numeric value.")
        except KeyboardInterrupt:
            print("\n\nPrediction cancelled by user.")
            sys.exit(0)


def collect_song_features(features):
    """Collect song features from user input."""
    print("\n" + "="*60)
    print("Enter song audio features")
    print("="*60)
    print("Note: Enter numeric values for each feature.")
    print("You can find these features from Spotify API or similar sources.\n")
    
    # Feature descriptions
    feature_info = {
        'danceability': '(0-1, how suitable for dancing)',
        'energy': '(0-1, intensity and activity)',
        'key': '(0-11, pitch class)',
        'loudness': '(dB, typically -60 to 0)',
        'mode': '(0=minor, 1=major)',
        'speechiness': '(0-1, presence of spoken words)',
        'acousticness': '(0-1, acoustic vs electronic)',
        'instrumentalness': '(0-1, likelihood of no vocals)',
        'liveness': '(0-1, presence of audience)',
        'valence': '(0-1, musical positiveness)',
        'tempo': '(BPM, typically 60-200)',
        'duration_ms': '(milliseconds)',
        'time_signature': '(typically 3-7)'
    }
    
    song_data = {}
    
    for feature in features:
        info = feature_info.get(feature, '')
        song_data[feature] = get_feature_input(feature, info)
    
    return song_data


def predict_popularity(song_data, model_data):
    """Make predictions using the trained model."""
    # Create DataFrame with correct feature order
    df = pd.DataFrame([song_data], columns=model_data['features'])
    
    # Scale features
    X_scaled = model_data['scaler'].transform(df)
    
    # Make prediction
    prediction = model_data['model'].predict(X_scaled)[0]
    
    return prediction


def interpret_prediction(avg_rank, weeks_on_chart):
    """Provide interpretation of the predictions."""
    print("\n" + "="*60)
    print("POPULARITY PREDICTION RESULTS")
    print("="*60)
    
    print(f"\nPredicted Average Rank: {avg_rank:.1f}")
    print("  → Lower rank = Higher popularity (1 is best)")
    
    if avg_rank < 20:
        rank_interpretation = "Excellent! Top-tier performance expected 🌟"
    elif avg_rank < 40:
        rank_interpretation = "Very Good! Strong chart performance expected ✨"
    elif avg_rank < 60:
        rank_interpretation = "Good! Moderate chart success expected 👍"
    elif avg_rank < 80:
        rank_interpretation = "Fair. May achieve modest chart success"
    else:
        rank_interpretation = "Challenging. May struggle on charts"
    
    print(f"  Interpretation: {rank_interpretation}")
    
    print(f"\nPredicted Weeks on Chart: {weeks_on_chart:.1f} weeks")
    
    if weeks_on_chart > 20:
        weeks_interpretation = "Exceptional longevity! 🎵"
    elif weeks_on_chart > 10:
        weeks_interpretation = "Great staying power! 🎶"
    elif weeks_on_chart > 5:
        weeks_interpretation = "Good chart presence 🎼"
    elif weeks_on_chart > 2:
        weeks_interpretation = "Moderate chart life"
    else:
        weeks_interpretation = "Short chart duration"
    
    print(f"  Interpretation: {weeks_interpretation}")
    
    print("\n" + "="*60)


def display_example_values():
    """Display example values for reference."""
    print("\n" + "="*60)
    print("EXAMPLE VALUES (for reference)")
    print("="*60)
    print("""
A typical K-pop song might have values like:
  - danceability: 0.7 (high)
  - energy: 0.8 (energetic)
  - key: 5 (F major/D minor)
  - loudness: -5.0 (dB)
  - mode: 1 (major key)
  - speechiness: 0.05 (mostly singing)
  - acousticness: 0.1 (mostly electronic)
  - instrumentalness: 0.0 (has vocals)
  - liveness: 0.15 (studio recording)
  - valence: 0.6 (moderately positive)
  - tempo: 120 (BPM)
  - duration_ms: 210000 (3.5 minutes)
  - time_signature: 4 (4/4 time)
    """)


def main():
    """Main function for interactive prediction."""
    print("="*60)
    print("K-pop Song Popularity Predictor")
    print("="*60)
    print("\nThis tool predicts how well a K-pop song will perform on charts")
    print("based on its audio features.\n")
    
    # Load models
    print("Loading trained models...")
    try:
        avg_rank_model = load_model('avg_rank')
        weeks_model = load_model('weeks_on_chart')
        print("✓ Models loaded successfully!\n")
    except Exception as e:
        print(f"Error loading models: {e}")
        sys.exit(1)
    
    # Ask if user wants to see examples
    show_examples = input("Would you like to see example feature values? (y/n): ").strip().lower()
    if show_examples == 'y':
        display_example_values()
    
    # Collect song features (using features from avg_rank model)
    song_data = collect_song_features(avg_rank_model['features'])
    
    # Make predictions
    print("\n" + "="*60)
    print("Making predictions...")
    print("="*60)
    
    avg_rank_pred = predict_popularity(song_data, avg_rank_model)
    weeks_pred = predict_popularity(song_data, weeks_model)
    
    # Display results
    interpret_prediction(avg_rank_pred, weeks_pred)
    
    # Ask if user wants to predict another song
    print()
    another = input("Would you like to predict another song? (y/n): ").strip().lower()
    if another == 'y':
        print("\n\n")
        main()
    else:
        print("\nThank you for using the K-pop Popularity Predictor!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nPrediction cancelled by user. Goodbye!")
        sys.exit(0)
