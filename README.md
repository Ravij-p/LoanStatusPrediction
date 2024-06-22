
---

# Loan Approval Prediction

## Overview
This project involves predicting loan approval based on various features using a Support Vector Machine (SVM) classifier. The dataset used contains information about loan applicants and whether their loan was approved.

## Steps

### 1. Data Loading and Exploration
The dataset is loaded using pandas and an initial exploration is performed to understand its structure. This includes displaying the first few rows to get a glimpse of the data and checking for any missing values.

### 2. Handling Missing Values
Missing values in the dataset are handled by dropping rows with any missing values. This ensures that the dataset is clean and ready for model training.

### 3. Data Encoding
Categorical variables in the dataset, such as Gender, Married, Education, Self_Employed, Property_Area, and Loan_Status, are encoded into numerical values. This transformation is necessary because machine learning algorithms require numerical input.

- **Loan_Status** is converted to binary (1 for 'Y', 0 for 'N').
- **Dependents** with value '3+' are replaced with 4.
- Other categorical variables are also mapped to numerical values accordingly.

### 4. Data Visualization
Seaborn is used to create count plots for various features (Gender, Education, Married, Self_Employed, Property_Area) to visualize the distribution of loan statuses across these categories. This helps in understanding the relationship between different features and the target variable (Loan_Status).

### 5. Feature Selection
The features (independent variables) and the target variable (dependent variable) are separated. The Loan_ID column, which is not useful for prediction, is dropped. The rest of the features are used for training the model.

### 6. Data Splitting
The dataset is split into training and testing sets using sklearn's train_test_split function. This allows for evaluating the model's performance on unseen data. Stratified splitting ensures that both training and testing sets have a similar distribution of the target variable.

### 7. Model Training
An SVM classifier with a linear kernel is trained using the training data. SVM is chosen for its effectiveness in high-dimensional spaces and robustness in handling overfitting.

### 8. Model Evaluation
The accuracy of the model is evaluated on both the training and testing sets. This helps in understanding how well the model has learned from the training data and how it performs on new, unseen data.

### 9. Making Predictions
The script includes functionality to make predictions on new input data. It processes the new data in the same way as the training data (encoding categorical variables) and uses the trained SVM model to predict whether a loan will be approved or not.

### 10. Output
The script outputs the accuracy of the model on the training and testing sets and provides predictions for new input data. This includes displaying whether a loan is predicted to be approved or not based on the input features.

---

## How to Run

1. Ensure you have Python installed.
2. Install the required libraries using the command:
    ```
    pip install numpy pandas matplotlib seaborn scikit-learn
    ```
3. Load the dataset into the script.
4. Run the script to train the model and make predictions.

---

## Results

- The model achieves a certain accuracy on the training and testing datasets.
- Predictions can be made on new data inputs regarding loan approval.

---

This project demonstrates the complete workflow from data preprocessing, visualization, model training, evaluation, to making predictions using machine learning techniques.

---
