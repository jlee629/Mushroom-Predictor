# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 21:14:36 2024

@author: Jungyu Lee
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

mushroom_dataset = pd.read_csv('mushroom_data.csv')

mushroom_dataset.isnull().sum()

mushroom_dataset['stem-root'].value_counts()
mushroom_dataset['veil-type'].value_counts()

mushroom_dataset['stem-root'].fillna('unknown', inplace=True)
mushroom_dataset['veil-type'].fillna('unknown', inplace=True)

mushroom_dataset.isnull().sum()

X = mushroom_dataset.drop(['class'], axis=1)  
y = mushroom_dataset['class']

mushroom_dataset.dtypes

numeric_X = X.select_dtypes(include=['float64']).columns
categorical_X = X.select_dtypes(include=['object']).columns

numeric_X_transformer = Pipeline([
    ('scaler', StandardScaler())
    ])

categorical_transformer = Pipeline([
    ('onehot', OneHotEncoder())
    ])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_X_transformer, numeric_X),
        ('cat', categorical_transformer, categorical_X)
    ])

X_processed = preprocessor.fit_transform(X)
y_processed = OneHotEncoder(sparse=False).fit_transform(np.array(y).reshape(-1, 1))

X_train, X_test, y_train, y_test = train_test_split(X_processed, y_processed, test_size=0.2, random_state=21)

# neural network structure
input_size = X_train.shape[1]
hidden_layer_size = 36
output_size = y_train.shape[1]

# weight initialization
W1 = np.random.uniform(-1, 1, (input_size, hidden_layer_size))
W2 = np.random.uniform(-1, 1, (hidden_layer_size, output_size))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def predict(X, W1, W2):
    hidden_layer_input = np.dot(X, W1)
    hidden_layer_output = sigmoid(hidden_layer_input)
    final_input = np.dot(hidden_layer_output, W2)
    final_output = sigmoid(final_input)
    
    # predictions = np.argmax(final_output, axis=1)
    return final_output

def backpropagation(X, y, W1, W2, lr, epochs, batch_size):
    for epoch in range(epochs):
        # shuffle the dataset
        permutation = np.random.permutation(X.shape[0])
        X_shuffled = X[permutation]
        y_shuffled = y[permutation]
        
        for i in range(0, X.shape[0], batch_size):
            X_mini = X_shuffled[i:i+batch_size]
            y_mini = y_shuffled[i:i+batch_size]
            
            # forward propagation
            hidden_layer_input = np.dot(X_mini, W1)
            hidden_layer_output = sigmoid(hidden_layer_input)
            final_input = np.dot(hidden_layer_output, W2)
            final_output = sigmoid(final_input)
            
            error = y_mini - final_output
            
            # backward propagation
            delta_output = error * sigmoid_derivative(final_output)
            dW2 = np.dot(hidden_layer_output.T, delta_output) / batch_size  
            error_hidden_layer = np.dot(delta_output, W2.T) * sigmoid_derivative(hidden_layer_output)
            dW1 = np.dot(X_mini.T, error_hidden_layer) / batch_size
            
            W2 += lr * dW2
            W1 += lr * dW1
                
    return W1, W2

# hyperparameters
lr = 0.1
epochs = 100
batch_size=32

W1, W2 = backpropagation(X_train, y_train, W1, W2, lr, epochs, batch_size)

y_pred = predict(X_test, W1, W2)

predicted_labels = np.argmax(y_pred, axis=1)
actual_labels = np.argmax(y_test, axis=1)

test_accuracy = accuracy_score(actual_labels, predicted_labels)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

for index in range(5):    
    print(f"Input= {X_test[index]}, \nActual = {np.argmax(y_test[index])}, \nPredicted= {np.argmax(y_pred[index])}")

