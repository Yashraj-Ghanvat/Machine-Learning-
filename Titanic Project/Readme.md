# 🚢 Titanic Survival Prediction - Machine Learning Project

This project uses the **Titanic dataset** to predict passenger survival based on features such as age, sex, fare, and class using machine learning models.

The notebook covers the complete ML workflow — from data loading to model evaluation — with multiple classifiers such as:
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

---

## 📊 Dataset Overview

- **Dataset Used:** Titanic Dataset (CSV file loaded from local directory)
- **Target Variable:** `Survived` (1 = survived, 0 = not survived)
- **Features Used:**
  - Passenger Class (`Pclass`)
  - Name, Sex, Age
  - Siblings/Spouses (`SibSp`)
  - Parents/Children (`Parch`)
  - Fare, Embarked, etc.

---

## 📁 Project Structure

- `Titanic.ipynb`: Jupyter Notebook that includes:
  - Importing required libraries
  - Data loading and preprocessing
  - Feature selection and scaling
  - Model training and testing
  - Model performance comparison

---

## 🔧 Installation & Requirements

Install Python dependencies before running the notebook:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
