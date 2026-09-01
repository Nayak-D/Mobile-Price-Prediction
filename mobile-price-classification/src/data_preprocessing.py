"""
Data Preprocessing Module

This module handles data loading, cleaning, missing value handling,
outlier detection, and other data preparation tasks.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple, Optional


class DataPreprocessor:
    """Handles data loading and preprocessing tasks."""
    
    def __init__(self):
        """Initialize the preprocessor."""
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.target_column = 'price_range'
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            filepath: Path to the CSV file
            
        Returns:
            pandas DataFrame with loaded data
        """
        df = pd.read_csv(filepath)
        print(f"✓ Loaded data from {filepath}")
        print(f"  Shape: {df.shape}")
        return df
    
    def check_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Check and report missing values.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with missing value information
        """
        missing = pd.DataFrame({
            'Column': df.columns,
            'Missing_Count': df.isnull().sum().values,
            'Missing_Percentage': (df.isnull().sum().values / len(df) * 100).round(2)
        })
        missing = missing[missing['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
        
        if len(missing) > 0:
            print("Missing Values Found:")
            print(missing.to_string(index=False))
        else:
            print("✓ No missing values found!")
        
        return missing
    
    def handle_missing_values(self, df: pd.DataFrame, method: str = 'mean') -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df: Input DataFrame
            method: Method to handle missing values ('mean', 'median', 'drop')
            
        Returns:
            DataFrame with missing values handled
        """
        df_clean = df.copy()
        
        if method == 'mean':
            df_clean = df_clean.fillna(df_clean.mean(numeric_only=True))
        elif method == 'median':
            df_clean = df_clean.fillna(df_clean.median(numeric_only=True))
        elif method == 'drop':
            df_clean = df_clean.dropna()
        
        print(f"✓ Missing values handled using {method} method")
        return df_clean
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate rows from the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with duplicates removed
        """
        initial_len = len(df)
        df_clean = df.drop_duplicates()
        removed = initial_len - len(df_clean)
        
        if removed > 0:
            print(f"✓ Removed {removed} duplicate rows")
        else:
            print("✓ No duplicate rows found")
        
        return df_clean
    
    def detect_outliers(self, df: pd.DataFrame, method: str = 'iqr', threshold: float = 1.5) -> dict:
        """
        Detect outliers using IQR or Z-score method.
        
        Args:
            df: Input DataFrame
            method: Method to detect outliers ('iqr' or 'zscore')
            threshold: Threshold for outlier detection
            
        Returns:
            Dictionary with outlier information
        """
        outliers_info = {}
        
        if method == 'iqr':
            for col in df.select_dtypes(include=[np.number]).columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = df[(df[col] < (Q1 - threshold * IQR)) | (df[col] > (Q3 + threshold * IQR))].shape[0]
                if outliers > 0:
                    outliers_info[col] = outliers
        
        if outliers_info:
            print("Outliers Detected (IQR method):")
            for col, count in sorted(outliers_info.items(), key=lambda x: x[1], reverse=True):
                print(f"  {col}: {count} outliers")
        else:
            print("✓ No outliers detected")
        
        return outliers_info
    
    def handle_outliers(self, df: pd.DataFrame, method: str = 'iqr', action: str = 'remove') -> pd.DataFrame:
        """
        Handle outliers in the dataset.
        
        Args:
            df: Input DataFrame
            method: Detection method ('iqr' or 'zscore')
            action: Action to take ('remove' or 'cap')
            
        Returns:
            DataFrame with outliers handled
        """
        df_clean = df.copy()
        
        if method == 'iqr':
            for col in df_clean.select_dtypes(include=[np.number]).columns:
                if col == self.target_column:
                    continue
                    
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                if action == 'remove':
                    initial_len = len(df_clean)
                    df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
                    removed = initial_len - len(df_clean)
                    if removed > 0:
                        print(f"  Removed {removed} outliers from {col}")
                elif action == 'cap':
                    df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
        
        return df_clean
    
    def scale_features(self, X_train: pd.DataFrame, X_test: Optional[pd.DataFrame] = None, 
                      method: str = 'standard') -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Scale features using StandardScaler or MinMaxScaler.
        
        Args:
            X_train: Training features
            X_test: Test features (optional)
            method: Scaling method ('standard' or 'minmax')
            
        Returns:
            Tuple of scaled training and test features
        """
        if method == 'standard':
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()
        
        self.scaler = scaler
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        
        if X_test is not None:
            X_test_scaled = pd.DataFrame(
                scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
            print(f"✓ Features scaled using {method} scaler")
            return X_train_scaled, X_test_scaled
        
        print(f"✓ Features scaled using {method} scaler")
        return X_train_scaled, None
    
    def preprocess_pipeline(self, df: pd.DataFrame, handle_outliers: bool = False) -> pd.DataFrame:
        """
        Apply complete preprocessing pipeline.
        
        Args:
            df: Input DataFrame
            handle_outliers: Whether to handle outliers
            
        Returns:
            Preprocessed DataFrame
        """
        print("\n" + "="*50)
        print("Starting Data Preprocessing Pipeline")
        print("="*50)
        
        # Step 1: Remove duplicates
        df_clean = self.remove_duplicates(df)
        
        # Step 2: Check missing values
        self.check_missing_values(df_clean)
        
        # Step 3: Handle missing values
        df_clean = self.handle_missing_values(df_clean)
        
        # Step 4: Detect outliers
        self.detect_outliers(df_clean)
        
        # Step 5: Handle outliers if requested
        if handle_outliers:
            df_clean = self.handle_outliers(df_clean, action='remove')
        
        print("\n✓ Preprocessing pipeline completed!")
        print(f"  Final shape: {df_clean.shape}")
        print("="*50 + "\n")
        
        return df_clean


def preprocess_train_test_data(train_path: str, test_path: str, 
                               handle_outliers: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and preprocess train and test data.
    
    Args:
        train_path: Path to training data
        test_path: Path to test data
        handle_outliers: Whether to handle outliers
        
    Returns:
        Tuple of preprocessed train and test DataFrames
    """
    preprocessor = DataPreprocessor()
    
    train_df = preprocessor.load_data(train_path)
    test_df = preprocessor.load_data(test_path)
    
    train_df = preprocessor.preprocess_pipeline(train_df, handle_outliers=handle_outliers)
    test_df = preprocessor.preprocess_pipeline(test_df, handle_outliers=handle_outliers)
    
    return train_df, test_df
