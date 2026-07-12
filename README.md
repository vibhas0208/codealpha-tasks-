# codealpha-tasks-
Credit scoring model using machine learning 
# 💳 Credit Scoring Model using Machine Learning

## 📌 Overview
This project demonstrates how to build a **credit scoring model** using machine learning.  
It predicts whether a loan applicant is likely to **default** or **repay**, based on financial and demographic features such as age, income, loan amount, and credit history.

The model uses **Logistic Regression** as a baseline, with preprocessing steps like feature scaling and train-test splitting.  
You can easily extend it with advanced models like **Random Forest** or **XGBoost**.

---

## ⚙️ Features
- End-to-end pipeline: data loading → preprocessing → training → evaluation → prediction
- Baseline model: **Logistic Regression** (interpretable and widely used in credit scoring)
- Evaluation metrics: Confusion Matrix, Classification Report, ROC-AUC
- Example prediction for a new applicant

---

## 📂 Project Structure
📌 Requirements
*Python 3.8+

*pandas

*numpy

*scikit-learn

Example prediction 
new_applicant = pd.DataFrame({
    "age":[35],
    "income":[50000],
    "loan_amount":[15000],
    "credit_history":[1]   # 1 = good history, 0 = poor
})
