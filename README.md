# Customer Churn Prediction using Logistic Regression

Name: Ashmit Kumar

Registration Number: 23BCE10453

Application Number: IN26011531

Batch Number: 1A

Email ID: ashmit.23bce10453@vitbhopal.ac.in

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

## Methodology
1. **Dataset Exploration**: Imported the customer churn dataset and examined its features to distinguish numerical and categorical variables while identifying `Churn` as the prediction target.
2. **Data Preparation**:
   - Converted the `TotalCharges` column to a numeric format and filled missing values using the median.
   - Transformed the `Churn` column into binary values (`Yes` = 1, `No` = 0).
   - Applied One-Hot Encoding to categorical attributes with `drop_first=True` to prevent redundancy.
   - Scaled all input features using `StandardScaler` to improve the stability and performance of the Logistic Regression algorithm.
3. **Train-Test Split**: Separated the processed dataset into training (80%) and testing (20%) subsets using `train_test_split`.
4. **Model Training**: Built a Logistic Regression classifier with `max_iter=1000` and trained it on the standardized training data.
5. **Performance Assessment**: Measured the model's effectiveness using Accuracy, Precision, Recall, F1-Score, and visualized the results with a Confusion Matrix.

## Results
- **Accuracy:** 81.97%
- **Precision:** 68.31%
- **Recall:** 59.52%
- **F1-Score:** 0.6361

## Conclusion
The Logistic Regression model achieved a strong overall accuracy of 81.97% in predicting customer churn. Features such as customer tenure, contract duration, and monthly billing emerged as the most influential factors affecting churn behavior. However, because Logistic Regression assumes a linear relationship between features and the target, it may struggle to identify more complex patterns within the data. Ensemble-based approaches such as Random Forest or XGBoost could potentially provide better performance, particularly in improving recall for customers who are likely to churn.
