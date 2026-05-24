# A2 — Heart Disease Classifier

## What this project does
Binary classification model to predict presence of heart disease 
using clinical features. Target: 0 = no disease, 1 = disease.

## Dataset
UCI Heart Disease Dataset — 920 patients, 4 hospital sources 
(Cleveland, Hungary, Switzerland, VA Long Beach)

## Stack
Python · pandas · numpy · scikit-learn · matplotlib · seaborn

## Results

| Model | Accuracy | Disease Recall | AUC |
|---|---|---|---|
| **SVC (rbf)** | **0.81** | **0.91** | 0.8847 |
| LR (C=1.0) | 0.80 | 0.86 | 0.895 |
| DT depth=3 | 0.755 | 0.863 | 0.857 |
| DT default | 0.739 | 0.78 | 0.734 |

**Final model: SVC with rbf kernel**
Disease Recall of 0.91 — catches 93 of 102 actual disease patients.

## Key concepts covered
- Logistic Regression + Regularization (C parameter)
- Decision Trees + overfitting via max_depth tuning
- SVM with rbf kernel
- Precision / Recall / F1 / ROC-AUC
- Bias-variance tradeoff
- 5-fold Cross-validation
- Full sklearn Pipeline with ColumnTransformer

## Andrew Ng connection
Course 1 — Classification week. Every model here connects back 
to decision boundaries, cost functions, and regularization from 
the ML Specialization.
