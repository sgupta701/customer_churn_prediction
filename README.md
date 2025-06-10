# 📊 Customer Churn Prediction
> Predicting whether a customer is likely to leave (churn) a telecom company, using machine learning.

## 🧠 Project Summary

> In this project, we analyze customer data to predict churn — that is, whether a customer will cancel their subscription or not. This helps telecom companies take proactive steps to retain customers.

We clean and prepare the data, select important features, try different ML models, and finally evaluate the best one using proper metrics.

## 📁 Folder Structure

```
customer-churn-prediction/
│
├── data/
│   ├── Telco_Customer_Churn_Dataset.csv
│   ├── cleaned_churn_data.csv
│   ├── X_train.csv, X_test.csv
│   ├── y_train.csv, y_test.csv
│   ├── X_train_selected.csv
│
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_split_data.ipynb
│   ├── 03_feature_selection.ipynb
│   ├── 04_model_selection.ipynb
│   ├── 05_model_training.ipynb
│   ├── 06_model_evaluation.ipynb
│
├── visuals/
│   ├── Churn vs Tenure.png
│   ├── Churn_Distribution.png
│   ├── Feature Correlation with Churn.png
│   ├── Receiving Operating Characteristics(ROC) Curve.png
│   ├── Top 10 Feature Importance from Random Forest.png
|
├── models/
│   └── logistic_regression_model.pkl
│
├── README.md
└── requirements.txt

```

## 🔧Tech Stack


<p float="left"> 
<img alt="Python" src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" /> 
<img alt="Jupyter" src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=Jupyter&logoColor=white" /> 
<img alt="pandas" src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" /> 
<img alt="NumPy" src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" /> 
<img alt="Scikit-learn" src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn logoColor=white" /> 
<img alt="Matplotlib" src="https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white" /> 
<img alt="Seaborn" src="https://img.shields.io/badge/Seaborn-0C5A6B?style=for-the-badge&logo=seaborn&logoColor=white" /> 
</p>

## 🔍 Step-by-Step Process

### Task 1: Data Cleaning

```
- Loaded raw telecom churn data.
- Handled missing values and wrong data types.
- Converted total charges to numeric.
- Label-encoded binary columns like gender, Partner, etc.
- One-hot encoded multi-category columns like Contract, PaymentMethod.

📌 Final output saved: cleaned_churn_data.csv

```

### Task 2: Data Splitting

```
- Split the cleaned data into features (X) and target (y).
- Used train_test_split() with stratify=y to maintain churn ratio.
- Saved training and testing sets.

📌 Files saved: X_train.csv, X_test.csv, y_train.csv, y_test.csv
```

### Task 3: Feature Selection

```
- Used domain knowledge to pick important features: MonthlyCharges, tenure, Contract, InternetService, PaymentMethod, etc.
- Also used .corr() and feature_importances_ to validate relevance.
```

### Task 4: Model Selection
```
> Tried 5 ML models:

1. Logistic Regression 
2. Decision Tree
3. Random Forest
4. Gradient Boosting
5. Support Vector Machine

✅ Logistic Regression performed best:

Balanced accuracy and interpretability.
Better recall and precision for the churn class than others.
```

### Task 5: Model Training
```
> Used Pipeline with StandardScaler + LogisticRegression.
Trained on the training set (X_train, y_train).
Saved the trained model as logistic_regression_model.pkl.
```

### Task 6: Model Evaluation
```
> Evaluated the model on X_test.

Metrics:
Accuracy: 80.4%
Precision (Churn): 65%
Recall (Churn): 57%
F1-score (Churn): 61%
ROC-AUC: Calculated for probability predictions.
✅ Plotted ROC curve and confusion matrix.
```

## 📈 Results
Model predicts churned customers with decent precision and recall.
Good enough to help the company take pre-emptive action on at-risk customers.

## ▶️ How to Run

Clone the repo: https://github.com/sgupta701/customer_churn_prediction.git

Install requirements:

- pip install -r requirements.txt
- Run notebooks step by step inside notebooks/

