import pandas as pd
import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(data):
    """Preprocess the data for training."""
    data = data.copy()
    features = data.drop(columns=['adoption_rate', 'name', 'id'])
    labels = data['adoption_rate']

    # Standardize the features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    return features_scaled, labels

def train_model(data):
    """Train the model on the sustainable materials data."""
    features, labels = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Build the neural network model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    # Train the model
    model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test))

    # Make predictions
    predictions = model.predict(features)

    return model, predictions

def make_predictions(model, data):
    """Make predictions using the trained model."""
    features, _ = preprocess_data(data)
    predictions = model.predict(features)
    return predictions
