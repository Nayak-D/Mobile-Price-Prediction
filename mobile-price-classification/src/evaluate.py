"""
Evaluation Module

This module handles model evaluation and performance metrics.
Functions include:
- Classification metrics calculation
- Confusion matrix analysis
- ROC curve plotting
- Model comparison
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    roc_auc_score
)
from typing import Dict, Tuple, Any


class ModelEvaluator:
    """Class to handle model evaluation and metrics."""
    
    def __init__(self):
        """Initialize ModelEvaluator."""
        self.metrics = {}
        self.confusion_matrices = {}
        self.predictions = {}
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                         model_name: str = 'Model', average: str = 'weighted') -> Dict[str, float]:
        """
        Calculate multiple classification metrics.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            model_name (str): Name of the model
            average (str): Averaging method for multi-class
            
        Returns:
            Dict[str, float]: Dictionary of metrics
        """
        metrics = {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, average=average, zero_division=0),
            'Recall': recall_score(y_true, y_pred, average=average, zero_division=0),
            'F1-Score': f1_score(y_true, y_pred, average=average, zero_division=0)
        }
        
        self.metrics[model_name] = metrics
        
        print(f"\n=== Metrics for {model_name} ===")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")
        
        return metrics
    
    def classification_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                            target_names: list = None) -> str:
        """
        Generate detailed classification report.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            target_names (list): Names of target classes
            
        Returns:
            str: Classification report
        """
        report = classification_report(y_true, y_pred, target_names=target_names)
        print("\n=== Detailed Classification Report ===")
        print(report)
        return report
    
    def confusion_matrix_analysis(self, y_true: np.ndarray, y_pred: np.ndarray,
                                 model_name: str = 'Model') -> np.ndarray:
        """
        Calculate and analyze confusion matrix.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            model_name (str): Name of the model
            
        Returns:
            np.ndarray: Confusion matrix
        """
        cm = confusion_matrix(y_true, y_pred)
        self.confusion_matrices[model_name] = cm
        
        print(f"\n=== Confusion Matrix for {model_name} ===")
        print(cm)
        
        return cm
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                             model_name: str = 'Model',
                             save_path: str = None,
                             figsize: Tuple[int, int] = (8, 6)) -> None:
        """
        Plot confusion matrix heatmap.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            model_name (str): Name of the model
            save_path (str): Path to save the plot
            figsize (Tuple[int, int]): Figure size
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Confusion matrix plot saved to {save_path}")
        
        plt.close()
    
    def plot_class_distribution(self, y: np.ndarray,
                               title: str = 'Class Distribution',
                               save_path: str = None,
                               figsize: Tuple[int, int] = (8, 5)) -> None:
        """
        Plot target class distribution.
        
        Args:
            y (np.ndarray): Target variable
            title (str): Plot title
            save_path (str): Path to save the plot
            figsize (Tuple[int, int]): Figure size
        """
        unique, counts = np.unique(y, return_counts=True)
        
        plt.figure(figsize=figsize)
        plt.bar(unique, counts, color='steelblue', edgecolor='black')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.title(title)
        plt.xticks(unique)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Class distribution plot saved to {save_path}")
        
        plt.close()
    
    def compare_models(self, models_metrics: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """
        Compare performance across multiple models.
        
        Args:
            models_metrics (Dict[str, Dict[str, float]]): Dictionary of model metrics
            
        Returns:
            pd.DataFrame: Comparison table
        """
        comparison_df = pd.DataFrame(models_metrics).T.sort_values('Accuracy', ascending=False)
        
        print("\n=== Model Comparison ===")
        print(comparison_df)
        
        return comparison_df
    
    def plot_model_comparison(self, models_metrics: Dict[str, Dict[str, float]],
                             save_path: str = None,
                             figsize: Tuple[int, int] = (12, 6)) -> None:
        """
        Plot comparison of models across metrics.
        
        Args:
            models_metrics (Dict[str, Dict[str, float]]): Dictionary of model metrics
            save_path (str): Path to save the plot
            figsize (Tuple[int, int]): Figure size
        """
        comparison_df = pd.DataFrame(models_metrics).T
        
        fig, ax = plt.subplots(figsize=figsize)
        comparison_df.plot(kind='bar', ax=ax)
        plt.xlabel('Model')
        plt.ylabel('Score')
        plt.title('Model Comparison')
        plt.legend(loc='lower right')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Model comparison plot saved to {save_path}")
        
        plt.close()
    
    def save_results(self, results_dict: Dict[str, Any], filepath: str = 'outputs/results/model_results.csv') -> None:
        """
        Save evaluation results to CSV.
        
        Args:
            results_dict (Dict[str, Any]): Results dictionary
            filepath (str): Path to save results
        """
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Convert to DataFrame
        if isinstance(results_dict, dict) and all(isinstance(v, dict) for v in results_dict.values()):
            results_df = pd.DataFrame(results_dict).T
        else:
            results_df = pd.DataFrame(results_dict)
        
        results_df.to_csv(filepath)
        print(f"✓ Results saved to {filepath}")


def create_evaluation_report(y_true: np.ndarray, y_pred: np.ndarray,
                            model_name: str,
                            evaluator: ModelEvaluator = None) -> Dict[str, Any]:
    """
    Create a comprehensive evaluation report.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        model_name (str): Name of the model
        evaluator (ModelEvaluator): ModelEvaluator instance
        
    Returns:
        Dict[str, Any]: Comprehensive evaluation report
    """
    if evaluator is None:
        evaluator = ModelEvaluator()
    
    print(f"\n{'='*50}")
    print(f"EVALUATION REPORT: {model_name}")
    print(f"{'='*50}")
    
    # Calculate metrics
    metrics = evaluator.calculate_metrics(y_true, y_pred, model_name=model_name)
    
    # Classification report
    class_report = evaluator.classification_report(y_true, y_pred)
    
    # Confusion matrix
    cm = evaluator.confusion_matrix_analysis(y_true, y_pred, model_name)
    
    # Plot confusion matrix
    evaluator.plot_confusion_matrix(y_true, y_pred, model_name,
                                   save_path=f'outputs/figures/confusion_matrix_{model_name.replace(" ", "_")}.png')
    
    report = {
        'Model': model_name,
        'Metrics': metrics,
        'Classification_Report': class_report,
        'Confusion_Matrix': cm
    }
    
    return report
