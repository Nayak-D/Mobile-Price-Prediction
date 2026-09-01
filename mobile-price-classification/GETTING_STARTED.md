# Getting Started Guide

## Quick Setup (5 minutes)

### 1. **Create Virtual Environment**
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# or
source .venv/bin/activate       # macOS/Linux
```

### 2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 3. **Prepare Data**
```bash
python prepare_data.py
```

### 4. **Run Analysis**

**Option A: Interactive Notebook (Recommended)**
```bash
jupyter notebook notebooks/mobile_price_classification.ipynb
```
- Cells execute top-to-bottom
- Run all cells with `Kernel → Restart & Run All`

**Option B: Python Script**
```bash
python -c "from src.train import *; print('Ready to import modules')"
```

---

## Project Overview

This is a **complete machine learning project** with:
- ✓ 1,600 training samples
- ✓ 400 test samples
- ✓ 20 hardware features
- ✓ 4 price categories (Low, Medium, High, Very High)
- ✓ 96.5% accuracy SVM model

---

## Key Files

| File | Purpose |
|------|---------|
| `notebooks/mobile_price_classification.ipynb` | Interactive analysis (Start here!) |
| `src/data_preprocessing.py` | Data loading and cleaning functions |
| `src/feature_engineering.py` | Feature analysis and selection |
| `src/train.py` | Model training utilities |
| `src/evaluate.py` | Evaluation metrics and visualization |
| `models/best_model.pkl` | Trained SVM model (ready to use) |
| `outputs/` | Generated figures, results, and reports |

---

## Making Predictions

```python
import pickle
import pandas as pd

# Load model
with open('models/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load scaler
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Prepare new data (must have same 20 features)
new_phones = pd.read_csv('new_data.csv')
new_phones_scaled = scaler.transform(new_phones)

# Predict
predictions = model.predict(new_phones_scaled)

# Map to labels
labels = {0: 'Low', 1: 'Medium', 2: 'High', 3: 'Very High'}
for pred in predictions:
    print(f"Price Category: {labels[pred]}")
```

---

## Project Structure

```
mobile-price-classification/
├── data/raw/                 ← Raw dataset (train.csv, test.csv)
├── data/processed/           ← Processed data
├── notebooks/                ← Jupyter notebooks (Analysis)
├── src/                      ← Python modules (Production code)
├── models/                   ← Trained models (best_model.pkl)
├── outputs/                  ← Results and visualizations
│   ├── figures/              ← PNG charts
│   └── results/              ← CSV reports
├── DOCUMENT/                 ← Project documentation
├── requirements.txt          ← Python dependencies
├── README.md                 ← Full documentation
└── .gitignore                ← Git ignore rules
```

---

## Common Tasks

### View Results
```bash
# See all generated outputs
ls outputs/figures/          # View generated charts
ls outputs/results/          # View result reports
```

### Check Model Performance
```bash
# Open and read the results CSV
cat outputs/results/model_results.csv
```

### Understand Data
```python
import pandas as pd
df = pd.read_csv('data/raw/train.csv')
print(df.head())
print(df.describe())
print(df.info())
```

### Train Custom Model
```python
from sklearn.ensemble import RandomForestClassifier
from src.data_preprocessing import load_data, preprocess_pipeline

# Load and preprocess
train_df = load_data('data/raw/train.csv')
train_df = preprocess_pipeline(train_df)

# Split features and target
X = train_df.drop('price_range', axis=1)
y = train_df['price_range']

# Train model
model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X, y)

# Save model
import pickle
with open('models/custom_model.pkl', 'wb') as f:
    pickle.dump(model, f)
```

---

## Troubleshooting

### Q: Import errors when running code?
**A:** Make sure virtual environment is activated and dependencies installed
```bash
.venv\Scripts\activate
pip install -r requirements.txt
```

### Q: Jupyter notebook won't start?
**A:** Ensure jupyter is installed
```bash
pip install jupyter ipython ipykernel
```

### Q: Data files not found?
**A:** Run data preparation script first
```bash
python prepare_data.py
```

### Q: What are the 20 features?
**A:** RAM, Battery Power, Clock Speed, N Cores, Primary Camera, Screen Width, Screen Height, Front Camera, Internal Memory, Mobile Weight, Dual SIM, 3G Support, 4G Support, WiFi Support, Bluetooth Support, Talk Time, Screen Height, Screen Width, Pixel Height, Pixel Width

### Q: How do I update the model?
**A:** Retrain with new data:
```python
# Load new data
new_df = pd.read_csv('new_training_data.csv')

# Preprocess
new_df = preprocess_pipeline(new_df)
X = new_df.drop('price_range', axis=1)
y = new_df['price_range']

# Train best model (SVM)
from sklearn.svm import SVC
svm = SVC(kernel='rbf', C=10, gamma='scale')
svm.fit(X, y)

# Save
with open('models/best_model.pkl', 'wb') as f:
    pickle.dump(svm, f)
```

---

## Next Steps

1. **Explore the Data**
   - Run the Jupyter notebook
   - Review visualizations in `outputs/figures/`

2. **Understand the Models**
   - Read the model comparison results
   - Check why SVM performs best

3. **Make Predictions**
   - Use the saved model on new data
   - Integrate into your application

4. **Improve the Model**
   - Collect more data
   - Engineer new features
   - Tune hyperparameters
   - Try ensemble methods

5. **Deploy to Production**
   - Create API (Flask/FastAPI)
   - Containerize (Docker)
   - Deploy to cloud (AWS/GCP/Azure)

---

## Learning Resources

- **Python**: https://www.python.org/
- **Pandas**: https://pandas.pydata.org/docs/
- **Scikit-learn**: https://scikit-learn.org/stable/documentation.html
- **Jupyter**: https://jupyter.org/
- **ML Basics**: https://developers.google.com/machine-learning/crash-course

---

## Support

For questions or issues:
1. Check this guide
2. Review the full README.md
3. Read the project report in DOCUMENT/
4. Check error messages carefully
5. Review the notebook examples

---

**Happy Learning! 🚀**

Start with: `jupyter notebook notebooks/mobile_price_classification.ipynb`
