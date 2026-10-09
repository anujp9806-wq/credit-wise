# CreditWise — Loan Approval Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)

A machine learning project that predicts whether a loan application will be approved based on applicant financial and demographic data. Built with Python, scikit-learn, and pandas.

## 📌 Overview

**CreditWise** is a binary classification project that analyzes loan application data to predict approval outcomes. The project covers the full data science pipeline — from data cleaning and EDA to feature engineering, model training, and evaluation.

**Target Variable:** `Loan_Approved` (Yes / No)

## 📂 Dataset

- **File:** `loan_approval_data.csv`
- **Rows:** 1,000
- **Columns:** 20 (before encoding)
- **Missing Values:** 50 per column

| Category | Features |
|----------|----------|
| **Applicant Info** | Applicant_Income, Coapplicant_Income, Age, Gender, Marital_Status, Dependents, Education_Level, Employment_Status, Employer_Category |
| **Financial Info** | Credit_Score, Existing_Loans, DTI_Ratio, Savings, Collateral_Value |
| **Loan Info** | Loan_Amount, Loan_Term, Loan_Purpose, Property_Area |
| **Target** | Loan_Approved |

## 🛠️ Tech Stack

- **Python 3.13**
- **pandas** — data manipulation
- **NumPy** — numerical operations
- **Matplotlib / Seaborn** — visualization
- **scikit-learn** — preprocessing, modeling, evaluation

## 🔄 Project Pipeline

### 1. Data Cleaning
- Removed `Applicant_ID` (non-predictive identifier)
- Numerical columns → mean imputation
- Categorical columns → most-frequent imputation

### 2. Exploratory Data Analysis (EDA)
- Class balance check on `Loan_Approved`
- Distribution plots & box plots for outliers
- Correlation heatmap of numerical features

**Key Insights:**
- `Credit_Score` — strong **positive** correlation (+0.45)
- `DTI_Ratio` — strong **negative** correlation (−0.44)
- `Applicant_Income` — mild positive relationship (+0.12)
- Dataset imbalanced: 62.1% No / 37.9% Yes

### 3. Feature Encoding
- **Label Encoding:** `Education_Level`, `Loan_Approved`
- **One-Hot Encoding:** `Employment_Status`, `Loan_Purpose`, `Marital_Status`, `Property_Area`, `Employer_Category`, `Gender`

### 4. Feature Engineering
- `DTI_Ratio_sq = DTI_Ratio²`
- `Credit_Score_sq = Credit_Score²`

### 5. Train/Test Split & Scaling
- 80/20 split (`random_state=42`)
- Standardized with `StandardScaler`

### 6. Model Training & Evaluation

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| **Logistic Regression** | **87.5%** | 0.790 | 0.803 | **0.797** |
| Naive Bayes | 86.5% | 0.783 | 0.770 | 0.777 |
| K-Nearest Neighbors (k=5) | 75.5% | 0.620 | 0.508 | 0.559 |

**Best Model:** Logistic Regression (after feature engineering).

## 📊 Results Summary

- ✅ Best accuracy: **87.5%** (Logistic Regression with engineered features)
- 📈 Feature engineering improved Logistic Regression:
  - Accuracy: 86.5% → **87.5%**
  - F1 Score: 0.777 → **0.797**
- ⚠️ KNN struggled with high-dimensional one-hot encoded data


