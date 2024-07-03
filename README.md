# SMS Spam Detection Project

## Overview
This project aims to build a machine learning model for detecting SMS spam messages using the SMS Spam Collection dataset from Kaggle. The model utilizes Natural Language Processing techniques to classify messages as either spam or ham (non-spam).

## Dataset
The dataset used for this project is the [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) from Kaggle. It consists of SMS messages labeled as 'spam' or 'ham'.

## Methodology
1. **Data Preprocessing**:
   - Loaded the dataset and handled missing values.
   - Converted labels ('spam' and 'ham') to binary integers (0 for spam, 1 for ham).
   - Applied TF-IDF (Term Frequency-Inverse Document Frequency) vectorization to convert text data into numerical features.

2. **Model Training and Evaluation**:
   - Split the dataset into training and testing sets (80% training, 20% testing).
   - Used Logistic Regression as the classification model.
   - Evaluated the model using metrics such as Accuracy, Precision, Recall, and F1 Score.

## Code Example
Here's a Python script (`spam_detection.py`) that trains the model and evaluates its performance:

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
raw_mail_data = pd.read_csv('/spam.csv', encoding='latin1')

# Handle missing values
mail_data = raw_mail_data.where(pd.notnull(raw_mail_data), '')

# Convert labels to binary integers
mail_data.loc[mail_data['v1'] == 'spam', 'v1'] = 0
mail_data.loc[mail_data['v1'] == 'ham', 'v1'] = 1

# Extract features and labels
X = mail_data['v2']
Y = mail_data['v1']

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

# Feature extraction using TF-IDF
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

# Convert labels to integers
Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

# Initialize and train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train_features, Y_train)

# Make predictions
Y_pred = model.predict(X_test_features)

# Evaluate the model
accuracy = accuracy_score(Y_test, Y_pred)
precision = precision_score(Y_test, Y_pred)
recall = recall_score(Y_test, Y_pred)
f1 = f1_score(Y_test, Y_pred)

# Print evaluation metrics
print(f"Accuracy Score : {accuracy}")
print(f"Precision Score : {precision}")
print(f"Recall Score : {recall}")
print(f"F1 Score : {f1}")

# Results Analysis

Accuracy Score : 0.9623318385650225
Precision Score : 0.959
Recall Score : 0.9989583333333333
F1 Score : 0.9785714285714285
