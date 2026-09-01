"""
Feature Engineering Module

This module handles feature analysis, selection, and creation.
Functions include:
- Statistical analysis of features
- Feature correlation analysis
- Feature selection techniques
- Feature creation and transformation
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple


def analyze_feature_statistics(X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
    """
    Analyze statistical properties of features.
    
    Args:
        X (pd.DataFrame): Feature dataframe
        y (pd.Series): Target variable (optional)
        
    Returns:
        pd.DataFrame: Statistical summary
    """
    stats = X.describe().T
    stats['skewness'] = X.skew()
    stats['kurtosis'] = X.kurtosis()
    
    print("\n=== Feature Statistics ===")
    print(stats)
    
    return stats


def correlation_analysis(X: pd.DataFrame, y: pd.Series, top_n: int = 10) -> pd.DataFrame:
    """
    Analyze correlation between features and target variable.
    
    Args:
        X (pd.DataFrame): Feature dataframe
        y (pd.Series): Target variable
        top_n (int): Number of top correlated features to show
        
    Returns:
        pd.DataFrame: Correlation values sorted by absolute value
    """
    # Create a temporary dataframe with features and target
    temp_df = X.copy()
    temp_df['target'] = y
    
    # Calculate correlation with target
    correlations = temp_df.corr()['target'].drop('target').abs().sort_values(ascending=False)
    
    print(f"\n=== Top {top_n} Features by Correlation with Target ===")
    print(correlations.head(top_n))
    
    return correlations


def select_features_by_correlation(X: pd.DataFrame, y: pd.Series, 
                                  threshold: float = 0.1) -> List[str]:
    """
    Select features based on correlation threshold.
    
    Args:
        X (pd.DataFrame): Feature dataframe
        y (pd.Series): Target variable
        threshold (float): Correlation threshold
        
    Returns:
        List[str]: List of selected feature names
    """
    correlations = correlation_analysis(X, y)
    selected_features = correlations[correlations >= threshold].index.tolist()
    
    print(f"\n✓ Selected {len(selected_features)} features with correlation >= {threshold}")
    return selected_features


def select_features_kbest(X: pd.DataFrame, y: pd.Series, k: int = 10, 
                          score_func = f_classif) -> Tuple[List[str], np.ndarray]:
    """
    Select top k features using SelectKBest.
    
    Args:
        X (pd.DataFrame): Feature dataframe
        y (pd.Series): Target variable
        k (int): Number of features to select
        score_func: Scoring function
        
    Returns:
        Tuple[List[str], np.ndarray]: Selected feature names and scores
    """
    selector = SelectKBest(score_func=score_func, k=min(k, X.shape[1]))
    X_selected = selector.fit_transform(X, y)
    
    selected_features = X.columns[selector.get_support()].tolist()
    scores = selector.scores_
    
    # Sort by score
    feature_scores = pd.DataFrame({
        'Feature': X.columns,
        'Score': scores
    }).sort_values('Score', ascending=False)
    
    print(f"\n=== Top {k} Features by SelectKBest ===")
    print(feature_scores.head(k))
    
    return selected_features, scores


def analyze_feature_importance(importance_dict: dict, top_n: int = 15) -> pd.DataFrame:
    """
    Analyze and visualize feature importance from a model.
    
    Args:
        importance_dict (dict): Dictionary with feature names as keys and importance as values
        top_n (int): Number of top features to display
        
    Returns:
        pd.DataFrame: Feature importance sorted
    """
    importance_df = pd.DataFrame(
        list(importance_dict.items()),
        columns=['Feature', 'Importance']
    ).sort_values('Importance', ascending=False)
    
    print(f"\n=== Top {top_n} Features by Importance ===")
    print(importance_df.head(top_n))
    
    return importance_df


def create_interaction_features(X: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """
    Create interaction features from specified feature pairs.
    
    Args:
        X (pd.DataFrame): Feature dataframe
        features (List[str]): List of feature pairs to interact
        
    Returns:
        pd.DataFrame: Dataframe with interaction features added
    """
    X_interaction = X.copy()
    
    for i in range(len(features) - 1):
        for j in range(i + 1, len(features)):
            feat1, feat2 = features[i], features[j]
            if feat1 in X.columns and feat2 in X.columns:
                interaction_name = f"{feat1}_x_{feat2}"
                X_interaction[interaction_name] = X[feat1] * X[feat2]
    
    print(f"✓ Created interaction features")
    return X_interaction


def create_polynomial_features(X: pd.DataFrame, features: List[str], degree: int = 2) -> pd.DataFrame:
    """
    Create polynomial features from specified features.
    
    Args:
        X (pd.DataFrame): Feature dataframe
        features (List[str]): List of features to create polynomials
        degree (int): Degree of polynomial
        
    Returns:
        pd.DataFrame: Dataframe with polynomial features added
    """
    X_poly = X.copy()
    
    for feat in features:
        if feat in X.columns:
            for d in range(2, degree + 1):
                poly_name = f"{feat}_poly{d}"
                X_poly[poly_name] = X[feat] ** d
    
    print(f"✓ Created polynomial features (degree={degree})")
    return X_poly


def perform_feature_engineering(X: pd.DataFrame, y: pd.Series, 
                               select_top_k: int = None) -> Tuple[pd.DataFrame, List[str]]:
    """
    Complete feature engineering pipeline.
    
    Args:
        X (pd.DataFrame): Feature dataframe
        y (pd.Series): Target variable
        select_top_k (int): Number of top features to select (None to keep all)
        
    Returns:
        Tuple[pd.DataFrame, List[str]]: Engineered features and selected feature names
    """
    print("\n=== Starting Feature Engineering ===")
    
    # Analyze features
    analyze_feature_statistics(X, y)
    
    # Correlation analysis
    correlation_analysis(X, y)
    
    # Feature selection
    if select_top_k is not None:
        selected_features, _ = select_features_kbest(X, y, k=select_top_k)
        X_engineered = X[selected_features].copy()
    else:
        X_engineered = X.copy()
    
    print("\n✓ Feature engineering completed!")
    return X_engineered, X_engineered.columns.tolist()


def plot_correlation_heatmap(X: pd.DataFrame, y: pd.Series, figsize: Tuple[int, int] = (12, 10),
                             save_path: str = None) -> None:
    """
    Plot correlation heatmap between features.
    
    Args:
        X (pd.DataFrame): Feature dataframe
        y (pd.Series): Target variable
        figsize (Tuple[int, int]): Figure size
        save_path (str): Path to save the plot
    """
    # Create temporary dataframe with target
    temp_df = X.copy()
    temp_df['price_range'] = y
    
    # Create heatmap
    plt.figure(figsize=figsize)
    correlation_matrix = temp_df.corr()
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Correlation heatmap saved to {save_path}")
    
    plt.close()
