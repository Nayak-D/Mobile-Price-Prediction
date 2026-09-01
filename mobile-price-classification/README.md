# Mobile Price Classification - Machine Learning Project

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange?logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)
![License](https://img.shields.io/badge/License-MIT-blue)

## 📋 Overview

A complete machine learning project that predicts mobile phone price categories based on hardware specifications. This is a **multi-class classification problem** using supervised learning techniques.

The project demonstrates a professional ML workflow including data preprocessing, exploratory analysis, feature engineering, model training, evaluation, and deployment-ready code.

## 🎯 Problem Statement

Given a set of mobile phone specifications (RAM, battery, camera, display, etc.), predict which price category the phone belongs to:
- **0** → Low Cost (Budget)
- **1** → Medium Cost
- **2** → High Cost (Premium)
- **3** → Very High Cost (Flagship)

## 📊 Dataset

- **Total Samples**: 2,000 mobile phones
- **Training Samples**: 1,600
- **Test Samples**: 400
- **Features**: 20 hardware specifications
- **Target Variable**: `price_range` (4 classes)
- **Data Source**: Raw CSV files stored in `data/raw/`

### Dataset Features

| Feature | Description | Unit |
|---------|-------------|------|
| battery_power | Battery capacity | mAh |
| blue | Bluetooth support | Binary (0/1) |
| clock_speed | Processor speed | GHz |
| dual_sim | Dual SIM support | Binary (0/1) |
| fc | Front camera quality | Megapixels |
| four_g | 4G support | Binary (0/1) |
| int_memory | Internal storage | GB |
| m_dep | Mobile depth | cm |
| mobile_wt | Mobile weight | g |
| n_cores | Number of CPU cores | Count |
| pc | Primary camera quality | Megapixels |
| px_height | Screen height | Pixels |
| px_width | Screen width | Pixels |
| ram | RAM size | MB |
| sc_h | Screen height | cm |
| sc_w | Screen width | cm |
| talk_time | Battery talk time | Hours |
| three_g | 3G support | Binary (0/1) |
| touch_screen | Touchscreen support | Binary (0/1) |
| wifi | WiFi support | Binary (0/1) |

## 📁 Project Structure

```
mobile-price-classification/
├── .venv/                          # Virtual environment (Python dependencies)
│
├── data/
│   ├── raw/                        # Original dataset
│   │   ├── train.csv               # Training data (1600 samples)
│   │   └── test.csv                # Test data (400 samples)
│   └── processed/
│       └── processed_data.csv       # Preprocessed dataset
│
├── notebooks/
│   └── mobile_price_classification.ipynb  # Interactive analysis & results
│
├── src/                            # Source code modules
│   ├── __init__.py                 # Package initialization
│   ├── data_preprocessing.py       # Data loading & cleaning
│   ├── feature_engineering.py      # Feature analysis & selection
│   ├── train.py                    # Model training utilities
│   └── evaluate.py                 # Model evaluation metrics
│
├── models/
│   ├── best_model.pkl              # Trained SVM model
│   └── scaler.pkl                  # Feature scaler
│
├── outputs/
│   ├── figures/                    # Generated visualizations
│   │   ├── class_distribution.png
│   │   ├── correlation_heatmap.png
│   │   ├── feature_correlation.png
│   │   ├── model_comparison.png
│   │   ├── confusion_matrix.png
│   │   └── feature_importance.png
│   │
│   └── results/                    # Results & reports
│       ├── model_results.csv       # Model performance metrics
│       ├── classification_report.txt
│       └── project_summary.txt
│
├── DOCUMENT/                       # Project documentation
│   ├── Project_Report.docx         # Detailed project report
│   └── Project_Report.pdf          # PDF version
│
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── prepare_data.py                 # Data preparation script

```

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip or conda

### Installation

1. **Clone the repository** (or download the project)
```bash
cd mobile-price-classification
```

2. **Create and activate virtual environment**
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Prepare the data**
```bash
python prepare_data.py
```

### Running the Project

#### Option 1: Jupyter Notebook (Recommended for Analysis)
```bash
jupyter notebook notebooks/mobile_price_classification.ipynb
```

This will open the interactive notebook in your browser with:
- EDA and visualizations
- Model training and comparison
- Results and conclusions

#### Option 2: Python Scripts (For Production)
```bash
python train.py
```

## 🔍 Workflow Overview

### 1. **Data Loading** 
- Load training and test datasets
- Display basic statistics and info

### 2. **Exploratory Data Analysis (EDA)**
- Analyze target variable distribution
- Feature statistics and relationships
- Correlation analysis
- Identify outliers and missing values

### 3. **Data Preprocessing**
- Handle missing values
- Remove duplicates
- Detect and manage outliers
- Feature scaling using StandardScaler

### 4. **Feature Engineering**
- Analyze feature importance
- Correlation analysis with target
- Feature selection techniques
- Create additional features if needed

### 5. **Model Development**
Train and compare 7 different models:
- Logistic Regression
- Decision Tree
- Random Forest
- Extra Trees
- Support Vector Machine (SVM) ⭐ **Best**
- K-Nearest Neighbors (KNN)
- Gradient Boosting

### 6. **Model Evaluation**
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix Analysis
- Classification Report
- Model Comparison

### 7. **Results & Deployment**
- Save best model
- Generate visualizations
- Create documentation

## 📊 Results

### Best Model: **Support Vector Machine (SVM)**

| Metric | Value |
|--------|-------|
| Test Accuracy | **0.9650** (96.5%) |
| Precision | 0.9657 |
| Recall | 0.9650 |
| F1-Score | 0.9651 |

### Model Performance Comparison

| Model | Train Accuracy | Test Accuracy |
|-------|----------------|---------------|
| **SVM** | **0.9825** | **0.9650** |
| Gradient Boosting | 0.9937 | 0.9475 |
| Extra Trees | 0.9950 | 0.9475 |
| Random Forest | 0.9950 | 0.9425 |
| KNN | 0.9513 | 0.9325 |
| Decision Tree | 0.9994 | 0.9250 |
| Logistic Regression | 0.9406 | 0.9125 |

## 🛠️ Technologies Used

| Category | Technologies |
|----------|---------------|
| **Language** | Python 3.8+ |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | Scikit-Learn |
| **Visualization** | Matplotlib, Seaborn |
| **Notebooks** | Jupyter |
| **Development** | VS Code, Git |

## 📈 Generated Outputs

### Visualizations
- `class_distribution.png` - Target variable distribution
- `correlation_heatmap.png` - Feature correlation matrix
- `feature_correlation.png` - Top features by correlation
- `model_comparison.png` - Model performance comparison
- `confusion_matrix.png` - Best model confusion matrix
- `feature_importance.png` - Feature importance ranking

### Reports
- `model_results.csv` - Model performance metrics
- `classification_report.txt` - Detailed metrics per class
- `project_summary.txt` - Complete project summary

### Models
- `best_model.pkl` - Trained SVM model (ready for predictions)
- `scaler.pkl` - Feature scaler (for preprocessing new data)

## 💻 Using the Trained Model

### Making Predictions on New Data

```python
import pickle
import pandas as pd

# Load the trained model and scaler
with open('models/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load new data
new_data = pd.read_csv('new_phones.csv')

# Scale features (must use the saved scaler)
new_data_scaled = scaler.transform(new_data)

# Make predictions
predictions = model.predict(new_data_scaled)

# Convert predictions to labels
price_labels = {
    0: 'Low Cost',
    1: 'Medium Cost',
    2: 'High Cost',
    3: 'Very High Cost'
}

for i, pred in enumerate(predictions):
    print(f"Phone {i+1}: {price_labels[pred]}")
```

## 🔬 Feature Engineering Insights

### Top 10 Most Important Features (SVM)
1. RAM - Strong positive correlation (0.89)
2. Battery Power - Strong positive correlation (0.82)
3. Clock Speed - Moderate correlation (0.71)
4. N Cores - Moderate correlation (0.68)
5. Primary Camera - Moderate correlation (0.62)
6. Screen Width - Moderate correlation (0.58)
7. Screen Height - Moderate correlation (0.55)
8. 4G Support - Binary feature (0.42)
9. Front Camera - Weak correlation (0.35)
10. WiFi Support - Binary feature (0.28)

## 📚 Key Learnings

1. **RAM is the strongest predictor** of mobile price category
2. **Battery capacity and processor speed** are critical factors
3. **SVM with RBF kernel** outperforms other algorithms for this task
4. **Feature scaling** significantly improves model performance
5. **Class distribution is balanced**, reducing bias issues
6. **Hyperparameter tuning** can improve model performance further

## 🔧 Customization & Extension

### Hyperparameter Tuning
Modify model parameters in the notebook or training script:
```python
models = {
    'SVM': SVC(kernel='rbf', C=10, gamma='scale', random_state=42),  # Adjust C and gamma
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42),
}
```

### Cross-Validation
Implement k-fold cross-validation:
```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X_train, y_train, cv=5)
```

### Feature Selection
Reduce model complexity with feature selection:
```python
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(f_classif, k=15)
X_selected = selector.fit_transform(X_train, y_train)
```

## 📋 Project Checklist

- [x] Data loading and exploration
- [x] Exploratory Data Analysis (EDA)
- [x] Data preprocessing and cleaning
- [x] Feature engineering and analysis
- [x] Model training (7 algorithms)
- [x] Model evaluation and comparison
- [x] Best model selection
- [x] Visualization and reporting
- [x] Model saving for deployment
- [x] Documentation and README

## 🚀 Production Deployment

To deploy the model:

1. **API Development**
```bash
# Example using Flask
pip install flask
# Create api.py with Flask endpoint
```

2. **Containerization**
```bash
# Create Dockerfile for deployment
docker build -t mobile-price-classifier .
docker run -p 5000:5000 mobile-price-classifier
```

3. **Cloud Deployment**
- AWS SageMaker
- Google Cloud ML
- Azure ML Studio

## 📖 References & Resources

- [Scikit-learn Documentation](https://scikit-learn.org)
- [Pandas Documentation](https://pandas.pydata.org)
- [Machine Learning Best Practices](https://developers.google.com/machine-learning)
- [Classification Metrics Guide](https://en.wikipedia.org/wiki/Confusion_matrix)

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Author

**Mobile Price Classification Team**
- Created: 2024
- Last Updated: 2024

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Contact & Support

For questions or support:
- Create an issue in the repository
- Contact: [Your Email/Website]

## ⭐ Acknowledgments

- Dataset source: Mobile Price Classification dataset
- Inspired by Kaggle competitions
- Built with open-source ML tools

---

**Made with ❤️ for the ML Community**

Happy Learning! 🚀
