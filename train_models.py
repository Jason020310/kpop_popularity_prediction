"""
K-pop Popularity Prediction Model Trainer

This script trains machine learning models to predict K-pop song popularity metrics:
- avg_rank: Average ranking position on charts
- weeks_on_chart: Number of weeks a song stays on charts

The trained models are saved to disk for later use in predictions.
"""

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import os


def load_data(data_path='./dataset/final_kpop_dataset.csv'):
    """Load the K-pop dataset."""
    print(f"Loading data from {data_path}...")
    data = pd.read_csv(data_path)
    print(f"Dataset loaded: {data.shape[0]} rows, {data.shape[1]} columns")
    return data


def prepare_features(data, target_column):
    """Prepare features and target for model training."""
    # Define features (exclude non-predictive columns)
    exclude_columns = [
        'avg_rank', 'Artist', 'Artist_Id', 'Track_Title', 'Track_Id', 'weeks_on_chart'
    ]
    features = [col for col in data.columns if col not in exclude_columns]
    
    X = data[features]
    y = data[target_column]
    
    print(f"Features: {features}")
    print(f"Target: {target_column}")
    print(f"Dataset size: {len(X)} samples")
    
    return X, y, features


def train_and_evaluate_models(X_train, X_test, y_train, y_test, target_name):
    """Train multiple models and evaluate their performance."""
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5)
    }
    
    results = {}
    print(f"\n{'='*60}")
    print(f"Training models for {target_name}")
    print(f"{'='*60}")
    
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate metrics
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        results[model_name] = {
            'model': model,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_mae': test_mae,
            'predictions': y_pred_test
        }
        
        print(f"  Train MSE: {train_mse:.2f}")
        print(f"  Test MSE: {test_mse:.2f}")
        print(f"  Train R²: {train_r2:.4f}")
        print(f"  Test R²: {test_r2:.4f}")
        print(f"  Test MAE: {test_mae:.2f}")
    
    # Find best model based on test R² score
    best_model_name = max(results.keys(), key=lambda k: results[k]['test_r2'])
    print(f"\n{'='*60}")
    print(f"Best model for {target_name}: {best_model_name}")
    print(f"Test R² Score: {results[best_model_name]['test_r2']:.4f}")
    print(f"{'='*60}")
    
    return results, best_model_name


def save_model(model, scaler, features, target_name, model_name):
    """Save the trained model, scaler, and metadata to disk."""
    model_dir = './models'
    os.makedirs(model_dir, exist_ok=True)
    
    model_data = {
        'model': model,
        'scaler': scaler,
        'features': features,
        'target': target_name,
        'model_type': model_name
    }
    
    filename = f"{model_dir}/{target_name}_model.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\nModel saved to {filename}")


def plot_predictions(y_test, results, target_name):
    """Plot actual vs predicted values for all models."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, (model_name, result) in enumerate(results.items()):
        ax = axes[idx]
        y_pred = result['predictions']
        
        ax.scatter(y_test, y_pred, alpha=0.5, s=20)
        ax.plot([y_test.min(), y_test.max()], 
                [y_test.min(), y_test.max()], 
                'r--', lw=2, label='Perfect Prediction')
        
        ax.set_xlabel(f'Actual {target_name}')
        ax.set_ylabel(f'Predicted {target_name}')
        ax.set_title(f'{model_name}\nR² = {result["test_r2"]:.4f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'./models/{target_name}_predictions.png', dpi=150)
    print(f"Prediction plot saved to ./models/{target_name}_predictions.png")
    plt.close()


def main():
    """Main function to train and save models."""
    print("="*60)
    print("K-pop Popularity Prediction Model Trainer")
    print("="*60)
    
    # Load data
    data = load_data()
    
    # Train models for both targets
    targets = ['avg_rank', 'weeks_on_chart']
    
    for target in targets:
        print(f"\n\n{'#'*60}")
        print(f"# Training models for: {target}")
        print(f"{'#'*60}\n")
        
        # Prepare data
        X, y, features = prepare_features(data, target)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train and evaluate models
        results, best_model_name = train_and_evaluate_models(
            X_train_scaled, X_test_scaled, y_train, y_test, target
        )
        
        # Save the best model
        best_model = results[best_model_name]['model']
        save_model(best_model, scaler, features, target, best_model_name)
        
        # Plot predictions
        plot_predictions(y_test, results, target)
    
    print("\n" + "="*60)
    print("Training complete! Models saved to ./models/ directory")
    print("="*60)


if __name__ == "__main__":
    main()
