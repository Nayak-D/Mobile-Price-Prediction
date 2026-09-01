import os

os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier


def main():
    df = pd.read_csv('dataset.csv')
    X = df.drop('price_range', axis=1)
    y = df['price_range']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        'Logistic Regression': LogisticRegression(max_iter=5000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Extra Trees': ExtraTreesClassifier(random_state=42),
        'SVM': SVC(kernel='rbf', C=10, gamma='scale'),
        'KNN': KNeighborsClassifier(n_neighbors=3),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        print(f'{name}: {acc:.4f}')

    best_model_name = max(results, key=results.get)
    best_model = models[best_model_name]
    best_model.fit(X_train, y_train)
    y_pred_best = best_model.predict(X_test)
    final_accuracy = accuracy_score(y_test, y_pred_best)

    print(f'Best Model: {best_model_name}')
    print(f'Final Accuracy: {final_accuracy:.4f}')
    print(classification_report(y_test, y_pred_best))
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred_best))

    # Compare model performances
    labels = list(results.keys())
    scores = list(results.values())
    plt.figure(figsize=(10, 6))
    plt.bar(labels, scores, color=['#4C72B0', '#55A868', '#C44E52', '#8172B3', '#CCB974', '#64B5CD', '#8C8C8C'])
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Comparison')
    plt.xticks(rotation=35, ha='right')
    plt.ylim(0.5, 1.0)
    for i, v in enumerate(scores):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150)
    plt.close()

    # Feature importance plot if available
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        feature_names = list(X.columns)
        indices = np.argsort(importances)[::-1][:10]
        plt.figure(figsize=(10, 6))
        plt.title('Top 10 Feature Importances')
        plt.bar(range(10), importances[indices], align='center')
        plt.xticks(range(10), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=150)
        plt.close()

    print('Model comparison plot saved as model_comparison.png')
    print('Feature importance plot saved as feature_importance.png')
    print('Code by Naga Sai')


if __name__ == '__main__':
    main()
