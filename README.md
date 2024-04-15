# Income Prediction Model 

#Overview

This repository contains code for building a machine learning model to predict whether an individual will earn less than or equal to $50,000 or more than $50,000 annually. The model is trained on a dataset containing features such as age, education level, occupation, etc., and the target variable is the income level categorized as <=80K or >80K.

Dependencies

1. Python 3.x
2. NumPy
3. pandas
4. scikit-learn
5. seaborn

# Dataset

The dataset used for training and testing the model is located in the data 
(https://archive.ics.uci.edu/ml/datasets/census+income) for sake of simplicity we use modified version of it.It consists of the following columns:

Age: Age of the individual
work class: Type of individual employement
Education: Education level of the individual
Occupation: Occupation of the individual
Marital Status: Marital status of the individual
Relationship: Relationship status of the individual
Race: Race of the individual
Sex: Gender of the individual
Hours per Week: Number of hours worked per week
Income: Income level categorized as <=50K or >50K (target variable)
native-country: Country of the individual


 # Model Evaluation

The performance of the model can be evaluated using metrics such as accuracy, precision, recall, F1-score, and confusion matrix. 
# Model selection
When it comes to predicting binary outcomes, such as whether an individual will earn less than or equal to $50,000 or more than $50,000 annually, different machine learning models have different prediction scales. Here's a brief overview of the prediction scales for some common classifiers:

# Logistic Regression:
Prediction Scale: Probabilities between 0 and 1.
Interpretation: The predicted probability represents the likelihood of belonging to the positive class (earning > $50,000).

# Support Vector Machine (SVM):
Prediction Scale: Decision boundary.
Interpretation: SVM aims to find the hyperplane that best separates the two classes. Predictions are based on which side of the hyperplane the data point falls on.

# Naive Bayes:
Prediction Scale: Probabilities between 0 and 1.
Interpretation: Similar to logistic regression, Naive Bayes calculates the probability of belonging to each class based on the input features and predicts the class with the highest probability.

# Decision Tree:
Prediction Scale: Classes or discrete values.
Interpretation: Decision trees recursively split the feature space into regions, assigning a class or value to each region. Predictions are based on which region the data point falls into.
# Random Forest:
Prediction Scale: Classes or discrete values.
Interpretation: Random forest is an ensemble of decision trees. Predictions are made by averaging or voting the predictions of individual trees.
For binary classification tasks like the one you described, all these models can provide predictions that can be interpreted accordingly. Logistic regression and Naive Bayes directly provide probabilities, which can be thresholded to obtain class predictions. SVM, decision trees, and random forests provide class predictions directly.

Depending on your specific requirements and the nature of your data, you may choose one of these models based on their prediction scale and performance characteristics.
