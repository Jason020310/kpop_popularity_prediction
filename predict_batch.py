"""
Batch K-pop Popularity Prediction Tool

This script allows users to predict popularity for multiple songs by reading
from a CSV file containing song features.

Usage:
    python predict_batch.py input_songs.csv output_predictions.csv
    
Input CSV should contain columns for all required audio features:
    danceability, energy, key, loudness, mode, speechiness, acousticness,
    instrumentalness, liveness, valence, tempo, duration_ms, time_signature
"""

import sys
import os
import pickle
import pandas as pd
import argparse


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


def validate_input_data(df, required_features):
    """Validate that input data contains all required features."""
    missing_features = [f for f in required_features if f not in df.columns]
    
    if missing_features:
        print(f"Error: Input file is missing required features: {missing_features}")
        print(f"\nRequired features: {required_features}")
        sys.exit(1)
    
    # Check for null values
    null_counts = df[required_features].isnull().sum()
    if null_counts.any():
        print("Warning: Input file contains null values:")
        print(null_counts[null_counts > 0])
        print("\nRows with null values will be skipped.")
    
    return True


def predict_batch(input_file, output_file):
    """Predict popularity for multiple songs from a CSV file."""
    print("="*60)
    print("K-pop Batch Popularity Predictor")
    print("="*60)
    
    # Load models
    print("\nLoading trained models...")
    avg_rank_model = load_model('avg_rank')
    weeks_model = load_model('weeks_on_chart')
    print("✓ Models loaded successfully!")
    
    # Load input data
    print(f"\nLoading input data from {input_file}...")
    try:
        df = pd.read_csv(input_file)
        print(f"✓ Loaded {len(df)} songs")
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading input file: {e}")
        sys.exit(1)
    
    # Validate input data
    print("\nValidating input data...")
    validate_input_data(df, avg_rank_model['features'])
    print("✓ Input data validated")
    
    # Prepare features
    X = df[avg_rank_model['features']].copy()
    
    # Remove rows with null values
    valid_rows = ~X.isnull().any(axis=1)
    X_clean = X[valid_rows]
    
    if len(X_clean) < len(X):
        print(f"\nSkipped {len(X) - len(X_clean)} rows due to missing values")
    
    # Make predictions
    print(f"\nMaking predictions for {len(X_clean)} songs...")
    
    # Scale features once (both models use the same features and should use the same scaler)
    X_scaled = avg_rank_model['scaler'].transform(X_clean)
    
    avg_rank_predictions = avg_rank_model['model'].predict(X_scaled)
    weeks_predictions = weeks_model['model'].predict(X_scaled)

    
    # Create output dataframe
    output_df = df[valid_rows].copy()
    output_df['predicted_avg_rank'] = avg_rank_predictions
    output_df['predicted_weeks_on_chart'] = weeks_predictions
    
    # Add interpretation columns
    output_df['rank_interpretation'] = output_df['predicted_avg_rank'].apply(
        lambda x: 'Excellent' if x < 20 else 
                 'Very Good' if x < 40 else 
                 'Good' if x < 60 else 
                 'Fair' if x < 80 else 'Challenging'
    )
    
    output_df['weeks_interpretation'] = output_df['predicted_weeks_on_chart'].apply(
        lambda x: 'Exceptional' if x > 20 else 
                 'Great' if x > 10 else 
                 'Good' if x > 5 else 
                 'Moderate' if x > 2 else 'Short'
    )
    
    # Save results
    print(f"\nSaving predictions to {output_file}...")
    output_df.to_csv(output_file, index=False)
    print("✓ Predictions saved successfully!")
    
    # Display summary statistics
    print("\n" + "="*60)
    print("PREDICTION SUMMARY")
    print("="*60)
    print(f"\nTotal songs processed: {len(output_df)}")
    print(f"\nAverage Rank Predictions:")
    print(f"  Mean: {avg_rank_predictions.mean():.1f}")
    print(f"  Min: {avg_rank_predictions.min():.1f}")
    print(f"  Max: {avg_rank_predictions.max():.1f}")
    print(f"\nWeeks on Chart Predictions:")
    print(f"  Mean: {weeks_predictions.mean():.1f}")
    print(f"  Min: {weeks_predictions.min():.1f}")
    print(f"  Max: {weeks_predictions.max():.1f}")
    
    print("\n" + "="*60)
    print(f"Results saved to: {output_file}")
    print("="*60)


def create_example_input():
    """Create an example input CSV file for reference."""
    example_data = {
        'song_name': ['Example Song 1', 'Example Song 2', 'Example Song 3'],
        'danceability': [0.75, 0.65, 0.80],
        'energy': [0.85, 0.75, 0.90],
        'key': [5, 7, 3],
        'loudness': [-4.5, -5.2, -3.8],
        'mode': [1, 0, 1],
        'speechiness': [0.06, 0.08, 0.05],
        'acousticness': [0.12, 0.15, 0.10],
        'instrumentalness': [0.0, 0.0, 0.0],
        'liveness': [0.15, 0.12, 0.18],
        'valence': [0.68, 0.55, 0.72],
        'tempo': [125, 110, 130],
        'duration_ms': [210000, 195000, 225000],
        'time_signature': [4, 4, 4]
    }
    
    df = pd.DataFrame(example_data)
    df.to_csv('example_input.csv', index=False)
    print("✓ Created example_input.csv")


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Batch prediction of K-pop song popularity',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict from a CSV file
  python predict_batch.py songs.csv predictions.csv
  
  # Create an example input file
  python predict_batch.py --example
        """
    )
    
    parser.add_argument('input_file', nargs='?', 
                       help='Input CSV file with song features')
    parser.add_argument('output_file', nargs='?', 
                       help='Output CSV file for predictions')
    parser.add_argument('--example', action='store_true',
                       help='Create an example input CSV file')
    
    args = parser.parse_args()
    
    if args.example:
        print("Creating example input file...")
        create_example_input()
        print("\nYou can now use this file as a template:")
        print("  python predict_batch.py example_input.csv predictions.csv")
        return
    
    if not args.input_file or not args.output_file:
        parser.print_help()
        print("\nError: Both input_file and output_file are required")
        print("Use --example to create a sample input file")
        sys.exit(1)
    
    predict_batch(args.input_file, args.output_file)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user. Goodbye!")
        sys.exit(0)
