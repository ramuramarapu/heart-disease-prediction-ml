# heart-disease-prediction-ml
Heart Disease Prediction using Machine Learning algorithms like Logistic Regression, Random Forest and SVM.
# Heart Disease Prediction using Machine Learning

## Project Description
This project predicts whether a patient has heart disease using machine learning algorithms.

## Technologies Used
- Python
- Machine Learning
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

## Dataset
Heart Disease Dataset from UCI Machine Learning Repository.

## Algorithms Used
- Logistic Regression
- Random Forest
- Support Vector Machine

## Output
The model predicts the probability of heart disease based on medical parameters.
python code:
# Heart Disease Prediction using Machine Learning

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("heart.csv")

# Split input and output
X = data.drop("target", axis=1)
y = data["target"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = LogisticRegression()

# Train model
model.fit(X_train, y_train)

# Prediction
prediction = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, prediction)
print("Model Accuracy:", accuracy)

# Example prediction
sample = np.array([[52,1,2,120,240,0,1,150,0,1.0,2,0,2]])
result = model.predict(sample)

if result[0] == 1:
    print("Person has Heart Disease")
else:
    print("Person is Healthy")
