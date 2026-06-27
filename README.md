# Customer Churn Prediction System

A supervised machine learning project that predicts whether a telecom customer is likely to churn, built in Python using Google Colab.

---

## Project Overview

This project uses the IBM Watson Telco Customer Churn dataset to build a binary classification system. It covers the complete machine learning workflow from data loading and preprocessing to model evaluation, feature importance analysis, and an interactive prediction interface.

The system compares Logistic Regression and Random Forest classifiers, automatically selects the best model, and saves it for reuse. A `predict_customer()` function and an ipywidgets-based UI allow instant predictions from new customer data.

---

## Features

- Exploratory data analysis with class distribution chart and correlation heatmap
- Data cleaning including TotalCharges type conversion and missing value handling
- Preprocessing pipeline using ColumnTransformer with OneHotEncoding
- Training and comparison of Logistic Regression and Random Forest models
- Evaluation using Accuracy, Precision, Recall, F1 Score, and Classification Report
- Side-by-side confusion matrix visualization for both models
- 5-fold cross-validation for performance consistency check
- Top 15 feature importances extracted from Random Forest
- Automatic best model selection based on test accuracy
- Model persistence with joblib (.pkl files)
- `predict_customer()` function accepting a customer dictionary
- Interactive prediction UI using ipywidgets inside Google Colab

---

## Dataset

| Property | Details |
|---|---|
| Name | WA_Fn-UseC_-Telco-Customer-Churn.csv |
| Source | IBM Watson Telco Customer Churn |
| Rows | 7,043 |
| Columns | 21 (20 features + 1 target) |
| Target Column | Churn (Yes / No) |
| Class Distribution | No: ~73.5% / Yes: ~26.5% |
| Missing Values | TotalCharges: 11 rows (whitespace-encoded) |

---

## Technologies Used

| Library | Version | Purpose |
|---|---|---|
| Python | 3.x | Core language |
| pandas | latest | Data loading and manipulation |
| numpy | latest | Numerical operations |
| matplotlib | latest | Charts and plots |
| seaborn | latest | Correlation heatmap |
| scikit-learn | latest | Models, preprocessing, evaluation |
| joblib | latest | Model saving and loading |
| ipywidgets | latest | Interactive prediction UI |

---

## Machine Learning Workflow

1. Load dataset from CSV
2. Inspect shape, head, info, and missing values
3. Exploratory Data Analysis (class distribution, correlation heatmap)
4. Drop `customerID`; convert `TotalCharges` to numeric; fill 11 missing values with median
5. Label encode `Churn` column
6. Split features into numerical and categorical; build ColumnTransformer
7. Train-test split: 80% train (5,634 samples) / 20% test (1,409 samples), stratified
8. Train Logistic Regression and Random Forest inside scikit-learn Pipelines
9. Evaluate both models; generate comparison table, bar chart, and confusion matrices
10. Run 5-fold cross-validation on both models
11. Plot top 15 feature importances from Random Forest
12. Auto-select best model; save `churn_model.pkl`, `preprocessor.pkl`, `label_encoder.pkl`
13. Define `predict_customer()` function; test with a sample customer
14. Launch ipywidgets interactive prediction interface

---

## Models Used

**Logistic Regression**
- solver: default (lbfgs)
- max_iter: 1000
- random_state: 42

**Random Forest Classifier**
- n_estimators: 100
- random_state: 42

Both models are wrapped inside scikit-learn Pipelines with the ColumnTransformer preprocessor, ensuring the same transformations are applied at training and prediction time.

---

## Model Evaluation

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Logistic Regression | 0.8048 | 0.6552 | 0.5588 | 0.6032 |
| Random Forest | 0.7779 | 0.6034 | 0.4759 | 0.5321 |

**5-Fold Cross-Validation**

| Model | CV Mean Accuracy | Std Deviation |
|---|---|---|
| Logistic Regression | 0.8036 | +/- 0.0074 |
| Random Forest | 0.7876 | +/- 0.0120 |

---

## Best Performing Model

**Logistic Regression** was automatically selected as the best model with a test accuracy of **80.48%** and a cross-validation accuracy of **80.36%**.

It outperformed Random Forest on all four metrics and showed lower variance across folds, indicating more consistent generalization on this dataset.

Saved as: `churn_model.pkl`

---

## Project Structure

```
Customer-Churn-Prediction/
│
├── Customer_Churn_Analysis.ipynb       # Main notebook
├── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Dataset
├── churn_model.pkl                     # Saved best model pipeline
├── preprocessor.pkl                    # Saved ColumnTransformer
├── label_encoder.pkl                   # Saved LabelEncoder
├── requirements.txt                    # Python dependencies
└── README.md
```

---

## Screenshots

**Class Distribution Chart**

![Class Distribution](screenshots/class_distribution.png)

**Correlation Heatmap**

![Correlation Heatmap](screenshots/correlation_heatmap.png)

**Model Comparison Bar Chart**

![Model Comparison](screenshots/model_comparison.png)

**Confusion Matrices**

![Confusion Matrices](screenshots/confusion_matrices.png)

**Feature Importance Chart**

![Feature Importance](screenshots/feature_importance.png)

**Interactive Prediction UI**

![ipywidgets UI](screenshots/prediction_ui.png)

---

## Future Scope

- Deploy as a web application using Flask or Streamlit
- Apply SMOTE to handle class imbalance and improve churn recall
- Hyperparameter tuning with GridSearchCV
- Add SHAP values for per-customer prediction explanations
- Integrate with a live CRM or billing system via API

---

## Author

**Sanskar S. Gaikwad**
2nd Year, Computer Science Engineering
Internship: StarInfoSec

---

## License

This project is licensed under the MIT License.
