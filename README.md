# Mobile Price Prediction using Machine Learning

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![Status](https://img.shields.io/badge/Project-Completed-brightgreen)

A machine learning project that predicts the price category of a mobile phone based on hardware specifications such as RAM, battery capacity, camera quality, and display features.

The target variable is:

* price_range

The price categories are:

* 0 → Low Cost
* 1 → Medium Cost
* 2 → High Cost
* 3 → Very High Cost

---

# Project Overview

This project follows a complete supervised learning workflow for a multi-class classification task. The goal is to analyze the dataset, build several suitable models, compare their performance, and select the best-performing model for mobile price prediction.

---

# Workflow Followed

1. Understand the dataset and define the problem
2. Perform EDA and data preprocessing
3. Handle missing values, duplicates, and outliers where required
4. Apply feature analysis and selection where applicable
5. Train multiple suitable machine learning models
6. Compare model performance using evaluation metrics
7. Select the best model with justification
8. Produce final predictions and summarize results

---

# Dataset Features

| Feature | Description |
| --- | --- |
| battery_power | Battery capacity (mAh) |
| blue | Bluetooth support |
| clock_speed | Processor speed |
| dual_sim | Dual SIM support |
| fc | Front camera megapixels |
| four_g | 4G support |
| int_memory | Internal storage |
| mobile_wt | Mobile weight |
| n_cores | Number of CPU cores |
| pc | Primary camera megapixels |
| px_height | Screen height |
| px_width | Screen width |
| ram | RAM size |
| sc_h | Screen height |
| sc_w | Screen width |
| talk_time | Battery talk time |
| three_g | 3G support |
| touch_screen | Touchscreen support |
| wifi | WiFi support |

---

# Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* Jupyter Notebook / VS Code

---

# Models Evaluated

The project evaluates multiple classification models to determine the most effective one for this problem.

* Logistic Regression
* Decision Tree
* Random Forest
* Extra Trees
* Support Vector Machine (SVM)
* K-Nearest Neighbors (KNN)
* Gradient Boosting

---

# Best Model and Accuracy

The final selected model is:

* SVM (Support Vector Machine)

Verified accuracy on the current dataset:

* 96.75%

This is above the required threshold of 95% and is the best-performing model among those tested.

---

# Data Visualization Included

The analysis includes:

* model accuracy comparison chart
* feature importance visualization
* class distribution review
* evaluation metrics such as precision, recall, and F1-score

---

# Final Conclusion

This project successfully demonstrates a complete machine learning workflow for mobile price prediction. After comparing multiple models, the SVM classifier proved to be the most accurate and reliable model for the dataset. The model achieved 96.75% accuracy, which confirms strong predictive performance and reliable classification for price range prediction.

---

# Repository Structure

```bash
mobile-price-prediction-ml/
├── dataset.csv
├── mobliepricepredication.ipynb
├── README.md
├── run_mobile_price_model.py
├── model_comparison.png
├── feature_importance.png
├── .gitignore
├── LICENSE
```

---

# How to Run

Create a virtual environment and install dependencies:

```bash
python -m venv .venv311
.\.venv311\Scripts\activate
python -m pip install pandas numpy matplotlib seaborn scikit-learn
```

Run the notebook or script:

```bash
jupyter notebook mobliepricepredication.ipynb
```

Or:

```bash
python run_mobile_price_model.py
```

---

# Author

Code by Naga Sai
