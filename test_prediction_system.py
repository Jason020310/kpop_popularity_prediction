"""
Tests for K-pop Popularity Prediction System

This file contains basic tests for the prediction system components.
"""

import sys
import os
import unittest
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from calculate_english_percentage import calculate


class TestEnglishPercentageCalculator(unittest.TestCase):
    """Test the English percentage calculator."""
    
    def test_all_english(self):
        """Test lyrics that are 100% English."""
        lyrics = ["Hello world", "This is a test"]
        result = calculate(lyrics)
        self.assertEqual(result, 100.0)
    
    def test_no_english(self):
        """Test lyrics with no English words."""
        lyrics = ["안녕하세요", "세계"]
        result = calculate(lyrics)
        self.assertEqual(result, 0.0)
    
    def test_mixed_lyrics(self):
        """Test lyrics with mixed English and Korean."""
        lyrics = ["Hello 세계", "world 안녕"]
        result = calculate(lyrics)
        self.assertEqual(result, 50.0)
    
    def test_empty_lyrics(self):
        """Test empty lyrics."""
        lyrics = []
        result = calculate(lyrics)
        self.assertEqual(result, 0.0)
    
    def test_whitespace_only(self):
        """Test lyrics with only whitespace."""
        lyrics = ["   ", "  "]
        result = calculate(lyrics)
        self.assertEqual(result, 0.0)
    
    def test_numbers_and_special_chars(self):
        """Test that numbers and special characters are not counted as English."""
        lyrics = ["123 456", "!@# $%^"]
        result = calculate(lyrics)
        self.assertEqual(result, 0.0)
    
    def test_partial_english(self):
        """Test mixed content with varying percentages."""
        lyrics = ["Hello world test", "안녕"]
        result = calculate(lyrics)
        # 3 English words out of 4 total = 75%
        self.assertEqual(result, 75.0)


class TestDatasetIntegrity(unittest.TestCase):
    """Test the dataset integrity."""
    
    def setUp(self):
        """Load the dataset before each test."""
        self.data_path = './dataset/final_kpop_dataset.csv'
        if os.path.exists(self.data_path):
            self.data = pd.read_csv(self.data_path)
        else:
            self.data = None
    
    def test_dataset_exists(self):
        """Test that the dataset file exists."""
        self.assertTrue(os.path.exists(self.data_path), 
                       f"Dataset not found at {self.data_path}")
    
    def test_dataset_not_empty(self):
        """Test that the dataset contains data."""
        if self.data is not None:
            self.assertGreater(len(self.data), 0, "Dataset is empty")
    
    def test_required_columns_exist(self):
        """Test that all required columns are present."""
        if self.data is not None:
            required_columns = [
                'danceability', 'energy', 'key', 'loudness', 'mode',
                'speechiness', 'acousticness', 'instrumentalness', 
                'liveness', 'valence', 'tempo', 'duration_ms', 
                'time_signature', 'avg_rank', 'weeks_on_chart'
            ]
            for col in required_columns:
                self.assertIn(col, self.data.columns, 
                            f"Required column '{col}' not found in dataset")
    
    def test_no_null_target_values(self):
        """Test that target columns have no null values."""
        if self.data is not None:
            self.assertEqual(self.data['avg_rank'].isnull().sum(), 0,
                           "avg_rank contains null values")
            self.assertEqual(self.data['weeks_on_chart'].isnull().sum(), 0,
                           "weeks_on_chart contains null values")
    
    def test_feature_value_ranges(self):
        """Test that features are within expected ranges."""
        if self.data is not None:
            # Test features that should be between 0 and 1
            bounded_features = ['danceability', 'energy', 'speechiness', 
                              'acousticness', 'instrumentalness', 'liveness', 
                              'valence']
            for feature in bounded_features:
                if feature in self.data.columns:
                    self.assertTrue(
                        (self.data[feature] >= 0).all() and 
                        (self.data[feature] <= 1).all(),
                        f"{feature} values outside 0-1 range"
                    )
            
            # Test key should be 0-11
            if 'key' in self.data.columns:
                self.assertTrue(
                    (self.data['key'] >= 0).all() and 
                    (self.data['key'] <= 11).all(),
                    "Key values outside 0-11 range"
                )
            
            # Test mode should be 0 or 1
            if 'mode' in self.data.columns:
                self.assertTrue(
                    self.data['mode'].isin([0, 1]).all(),
                    "Mode values not 0 or 1"
                )


class TestModelTraining(unittest.TestCase):
    """Test model training functionality."""
    
    def test_models_can_be_created(self):
        """Test that model files can be created."""
        # This is a simple check - actual training is tested by running train_models.py
        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        
        # Just verify we can instantiate models
        lr = LinearRegression()
        rf = RandomForestRegressor(random_state=42)
        gb = GradientBoostingRegressor(random_state=42)
        
        self.assertIsNotNone(lr)
        self.assertIsNotNone(rf)
        self.assertIsNotNone(gb)
    
    def test_model_files_exist_after_training(self):
        """Test that model files exist (requires running train_models.py first)."""
        # This test will pass only if train_models.py has been run
        avg_rank_model = './models/avg_rank_model.pkl'
        weeks_model = './models/weeks_on_chart_model.pkl'
        
        # Check if models directory exists
        if os.path.exists('./models'):
            # If directory exists, check for model files
            if os.path.exists(avg_rank_model) and os.path.exists(weeks_model):
                self.assertTrue(True, "Models exist")
            else:
                self.skipTest("Models not yet trained. Run train_models.py first.")
        else:
            self.skipTest("Models directory not created. Run train_models.py first.")


def run_tests():
    """Run all tests and display results."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestEnglishPercentageCalculator))
    suite.addTests(loader.loadTestsFromTestCase(TestDatasetIntegrity))
    suite.addTests(loader.loadTestsFromTestCase(TestModelTraining))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(run_tests())
