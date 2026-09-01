"""
Training Module

This module handles model training and hyperparameter tuning.
Functions include:
- Training multiple models
- Hyperparameter tuning
- Cross-validation
- Model saving and loading
"""

import pandas as pd
import numpy as np
import pickle
import os
from typing import Dict, List, Tuple, Any
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score


class ModelTrainer:
    """Class to handle model training and management."""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize ModelTrainer.
        
        Args:
            random_state (int): Random state for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.results = {}
    
    def initialize_models(self) -> Dict[str, Any]:
        """
        Initialize a dictionary of models to train.
        
        Returns:
            Dict[str, Any]: Dictionary of model instances
        """
        self.models = {
            'Logistic Regression': LogisticRegression(
                max_iter=5000,
                random_state=self.random_state
            ),
            'Decision Tree': DecisionTreeClassifier(
                random_state=self.random_state
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state
            ),
            'Extra Trees': ExtraTreesClassifier(
                n_estimators=100,
                random_state=self.random_state
            ),
            'SVM': SVC(
                kernel='rbf',
                C=10,
                gamma='scale',
                random_state=self.random_state
            ),
            'KNN': KNeighborsClassifier(
                n_neighbors=3
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=self.random_state
            )
        }
        
        print("✓ Initialized 7 models for training")
        return self.models
    
    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series, 
                    verbose: bool = True) -> Dict[str, float]:
        """
        Train all initialized models.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            verbose (bool): Print training progress
            
        Returns:
            Dict[str, float]: Training accuracies for each model
        """
        print("\n=== Training Models ===")
        self.results = {}
        
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            y_pred_train = model.predict(X_train)
            train_acc = accuracy_score(y_train, y_pred_train)
            self.results[name] = train_acc
            
            if verbose:
                print(f"  {name}: Train Acc = {train_acc:.4f}")
        
        print("\n✓ All models trained successfully!")
        return self.results
    
    def evaluate_models(self, X_test: pd.DataFrame, y_test: pd.Series,
                       verbose: bool = True) -> Dict[str, float]:
        """
        Evaluate all trained models on test set.
        
        Args:
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
            verbose (bool): Print evaluation results
            
        Returns:
            Dict[str, float]: Test accuracies for each model
        """
        print("\n=== Evaluating Models on Test Set ===")
        test_results = {}
        
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            test_acc = accuracy_score(y_test, y_pred)
            test_results[name] = test_acc
            
            if verbose:
                print(f"  {name}: Test Acc = {test_acc:.4f}")
        
        return test_results
    
    def select_best_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[str, Any]:
        """
        Select the best model based on test accuracy.
        
        Args:
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
            
        Returns:
            Tuple[str, Any]: Best model name and model instance
        """
        test_results = self.evaluate_models(X_test, y_test, verbose=False)
        
        self.best_model_name = max(test_results, key=test_results.get)
        self.best_model = self.models[self.best_model_name]
        
        print(f"\n✓ Best Model: {self.best_model_name} (Test Acc = {test_results[self.best_model_name]:.4f})")
        return self.best_model_name, self.best_model
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> Dict[str, float]:
        """
        Perform cross-validation for all models.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target
            cv (int): Number of folds
            
        Returns:
            Dict[str, float]: Cross-validation scores for each model
        """
        print(f"\n=== Cross-Validation (k={cv}) ===")
        cv_results = {}
        
        for name, model in self.models.items():
            scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
            cv_results[name] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'scores': scores
            }
            print(f"  {name}: {scores.mean():.4f} (+/- {scores.std():.4f})")
        
        return cv_results
    
    def hyperparameter_tuning(self, X_train: pd.DataFrame, y_train: pd.Series,
                             model_name: str = None, params: Dict = None) -> Dict:
        """
        Perform hyperparameter tuning for a specific model.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            model_name (str): Name of model to tune (None for best model)
            params (Dict): Parameter grid for GridSearchCV
            
        Returns:
            Dict: Best parameters and best score
        """
        if model_name is None:
            model_name = self.best_model_name if self.best_model_name else 'Random Forest'
        
        model = self.models.get(model_name)
        
        if model is None or params is None:
            print(f"Model {model_name} not found or no parameters provided")
            return {}
        
        print(f"\n=== Hyperparameter Tuning for {model_name} ===")
        
        grid_search = GridSearchCV(model, params, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        print(f"  Best Params: {grid_search.best_params_}")
        print(f"  Best Score: {grid_search.best_score_:.4f}")
        
        # Update model with best parameters
        self.models[model_name] = grid_search.best_estimator_
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_
        }
    
    def save_model(self, model_name: str = None, filepath: str = 'models/best_model.pkl') -> str:
        """
        Save a trained model to disk.
        
        Args:
            model_name (str): Name of model to save (None for best model)
            filepath (str): Path to save the model
            
        Returns:
            str: Path to saved model
        """
        if model_name is None:
            model = self.best_model
            model_name = self.best_model_name
        else:
            model = self.models.get(model_name)
        
        if model is None:
            print(f"Model {model_name} not found")
            return None
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        
        print(f"✓ Model '{model_name}' saved to {filepath}")
        return filepath
    
    def load_model(self, filepath: str) -> Any:
        """
        Load a trained model from disk.
        
        Args:
            filepath (str): Path to the saved model
            
        Returns:
            Any: Loaded model
        """
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        
        print(f"✓ Model loaded from {filepath}")
        return model
    
    def get_results_summary(self) -> pd.DataFrame:
        """
        Get a summary of all training results.
        
        Returns:
            pd.DataFrame: Summary of model performance
        """
        if not self.results:
            print("No results available. Train models first.")
            return None
        
        summary = pd.DataFrame(
            list(self.results.items()),
            columns=['Model', 'Train_Accuracy']
        ).sort_values('Train_Accuracy', ascending=False)
        
        return summary
