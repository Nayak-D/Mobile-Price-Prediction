#!/usr/bin/env python3
"""
Prepare data by splitting the original dataset into train/test sets.
"""

import pandas as pd
import os
from sklearn.model_selection import train_test_split

# Read the original dataset
original_dataset_path = r'c:\Users\LENOVO\OneDrive\Documents\Mobile Price Prediction\mobile-price-prediction-ml\dataset.csv'
df = pd.read_csv(original_dataset_path)

# Split into train and test sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(
    df.drop('price_range', axis=1),
    df['price_range'],
    test_size=0.2,
    random_state=42,
    stratify=df['price_range']
)

# Reconstruct train and test dataframes
train_df = X_train.copy()
train_df['price_range'] = y_train

test_df = X_test.copy()
test_df['price_range'] = y_test

# Create output directories if they don't exist
os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

# Save train and test sets
train_df.to_csv('data/raw/train.csv', index=False)
test_df.to_csv('data/raw/test.csv', index=False)

# Save processed data (for now, it's the same as original)
df.to_csv('data/processed/processed_data.csv', index=False)

print(f"✓ Train set created: data/raw/train.csv ({len(train_df)} samples)")
print(f"✓ Test set created: data/raw/test.csv ({len(test_df)} samples)")
print(f"✓ Processed data created: data/processed/processed_data.csv ({len(df)} samples)")
