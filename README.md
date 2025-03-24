Artificial Neural Network for Customer Churn Prediction
This repository contains an implementation of an artificial neural network (ANN) using TensorFlow to predict customer churn based on financial behavior and demographics. This project leverages deep learning to interpret patterns that may not be immediately apparent through traditional statistical methods.

Project Description
The objective of this project is to use an ANN to accurately predict whether a bank's customers will leave within the next six months. The model processes data points like credit score, geography, gender, age, tenure, balance, and number of products used.

Features
Data Preprocessing: Transformation of categorical data, feature scaling for optimal neural network performance.
ANN Model: A sequential model with multiple dense layers using ReLU activation, and a sigmoid output layer for binary classification.
Model Training: Configured with the Adam optimizer and binary cross-entropy loss function.
Evaluation: Uses a confusion matrix and accuracy score to evaluate the model performance against the test dataset.

Technologies Used
TensorFlow: For building and training the neural network.
Pandas: For data manipulation and ingestion.
NumPy: For numerical operations.
Scikit-learn: For data preprocessing and model evaluation.
Python: All scripts are written in Python.

Getting Started
Prerequisites
Ensure you have Python and pip installed. You will need TensorFlow, Pandas, NumPy, and Scikit-learn.

Model Insights
The ANN is trained on a subset of the dataset and validated on another subset to ensure the model's generalizability. Initial tests show promising results, and further tuning and testing can enhance the model's accuracy and reliability.
