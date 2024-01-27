# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 18:01:25 2023

@author: techv
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# Load Fashion-MNIST dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Normalize the data
X_train = X_train / 255.0
X_test = X_test / 255.0

# Flatten the images for the unsupervised neural network
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Define the dimensions for the lower-dimensional representation
encoding_dim = 32  # You can adjust this dimension as needed

# Create an autoencoder model for dimensionality reduction
input_img = Input(shape=(784,))  # 28x28 = 784
encoded = Dense(encoding_dim, activation='relu')(input_img)
decoded = Dense(784, activation='sigmoid')(encoded)
autoencoder = Model(input_img, decoded)

autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Train the autoencoder
autoencoder.fit(X_train_flat, X_train_flat, epochs=10, batch_size=256, shuffle=True, validation_data=(X_test_flat, X_test_flat))

# Use the trained autoencoder to obtain the lower-dimensional representations
encoder = Model(input_img, encoded)
encoded_images = encoder.predict(X_test_flat)

# Apply t-SNE to reduce dimensionality to 2
tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(encoded_images)

# Visualize the results
plt.figure(figsize=(10, 8))
scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=y_test, cmap='tab10', s=10)
plt.legend(handles=scatter.legend_elements()[0], labels=range(10))
plt.show()