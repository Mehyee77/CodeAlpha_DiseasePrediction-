# CodeAlpha_DiseasePrediction

Heart disease prediction using ML classification models — CodeAlpha ML Internship Task 4

## Overview
This project trains and compares four machine learning models to predict the
presence of heart disease using patient clinical data (918 records, 11 features).

## Dataset
Heart Failure Prediction Dataset (918 records, 12 columns).
**Features:** Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS,
RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope.
**Target:** HeartDisease (1 = disease, 0 = normal).

## Workflow
1. Exploratory Data Analysis (EDA)
2. Data cleaning — handling disguised missing values (0s in RestingBP & Cholesterol)
3. Preprocessing — one-hot encoding + feature scaling
4. Model training — Logistic Regression, SVM, Random Forest, XGBoost
5. Evaluation — Accuracy, Precision, Recall, F1-Score, ROC-AUC

## Results
| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| Random Forest | 89.1% | 0.939 |
| SVM | 86.4% | 0.939 |
| XGBoost | 87.5% | 0.937 |
| Logistic Regression | 89.1% | 0.935 |

Random Forest was the best performer, with ~91% recall — critical for a medical
task where missing a diagnosis is the costliest error. Most predictive features:
ST_Slope, asymptomatic chest pain, and Oldpeak.

## Tech Stack
Python, pandas, NumPy, scikit-learn, XGBoost, matplotlib, seaborn

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Open and run `heart_disease_prediction.ipynb`
