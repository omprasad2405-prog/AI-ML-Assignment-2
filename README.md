# Customer Churn Prediction using Logistic Regression

Name: OM PRASAD

Registration Number: 23BCG10091

Application Number: IN26011530

Batch Number: 1A


## Objective
The aim of this project is to develop a Logistic Regression model capable of predicting whether a customer is likely to leave a telecommunications service based on demographic information and service-related attributes.

## Dataset Link
- [Kaggle: Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## Libraries Used
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `kaggle`

Methodology
The project follows a structured data science workflow divided into distinct tasks:

Data Understanding:

Loaded the dataset and examined the initial records.
Categorized features into numerical (e.g., tenure, MonthlyCharges), categorical (e.g., Contract, PaymentMethod), and identified the target variable (Churn).
Data Preprocessing:

Missing Values: Handled blank space entries in the TotalCharges column (associated with new customers) by coercing them to NaN and dropping the affected rows.
Feature Engineering/Encoding: Converted the target variable (Yes/No) into binary format (1/0). Applied One-Hot Encoding (pd.get_dummies()) to transform categorical text variables into machine-readable numerical formats.
Data Splitting: Split the dataset into an 80% training set and a 20% testing set to ensure objective evaluation.
Scaling: Applied Standard Scaling to continuous numerical variables to optimize the Logistic Regression algorithm's convergence and performance.
Model Development:

Instantiated a Logistic Regression model.
Trained (fitted) the model using the 80% training data.
Generated churn predictions on the unseen 20% test dataset.
Model Evaluation:

Evaluated the model against standard classification metrics: Accuracy, Precision, Recall, and F1-Score.
Analyzed the Confusion Matrix to understand the distribution of True Positives, True Negatives, False Positives, and False Negatives.

## Results
- **Accuracy:** 81.97%
- **Precision:** 68.31%
- **Recall:** 59.52%
- **F1-Score:** 0.6361

## Conclusion
The Logistic Regression model achieved a strong overall accuracy of 81.97% in predicting customer churn. Features such as customer tenure, contract duration, and monthly billing emerged as the most influential factors affecting churn behavior. However, because Logistic Regression assumes a linear relationship between features and the target, it may struggle to identify more complex patterns within the data. Ensemble-based approaches such as Random Forest or XGBoost could potentially provide better performance, particularly in improving recall for customers who are likely to churn.
