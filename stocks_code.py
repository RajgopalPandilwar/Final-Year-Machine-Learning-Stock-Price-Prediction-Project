# -*- coding: utf-8 -*-
"""
Stock Price Prediction using LSTM Deep Learning Model
=====================================================
Author: Rajgopal Pandilwar
Based on original work by Vatshayan

This script demonstrates stock price prediction using an LSTM-based
neural network. It loads Google stock price data, preprocesses it,
and trains a deep learning model to predict future prices.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ============================================================
# STEP 1: Data Preprocessing
# ============================================================

# Load the training dataset (Google stock prices)
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
print(f"Dataset shape: {dataset_train.shape}")
print(dataset_train.head())

# Extract the 'Open' price column as a numpy array
training_set = dataset_train.iloc[:, 1:2].values
print(f"Training set shape: {training_set.shape}")
print(training_set[0:5])

# Visualize the raw training data
plt.plot(training_set, color='red', label='Google Stock Price (Training Set)')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

# ============================================================
# STEP 2: Feature Scaling
# ============================================================
# Normalize the data to [0, 1] using MinMaxScaler for better LSTM performance
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = scaler.fit_transform(training_set)

# ============================================================
# STEP 3: Create Training Sequences
# ============================================================
# Use 60 timesteps to predict the next value
X_train = []
y_train = []

for i in range(60, len(training_set_scaled)):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

# Reshape for LSTM input: (samples, timesteps, features)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# ============================================================
# STEP 4: Build the LSTM Model
# ============================================================
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

model = Sequential()

# First LSTM layer with dropout for regularization
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))

# Second LSTM layer
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

# Third LSTM layer
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))

# Output layer
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# ============================================================
# STEP 5: Train the Model
# ============================================================
model.fit(X_train, y_train, epochs=100, batch_size=32)

print("Model training complete. Use the trained model for predictions.")
