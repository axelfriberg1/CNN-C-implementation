#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 13:04:52 2025

@author: axelfriberg
"""

import numpy as np
from tensorflow.keras.datasets import mnist

# Load MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Combine train + test (optional; remove if you only want train)
X = np.concatenate([x_train, x_test], axis=0)
y = np.concatenate([y_train, y_test], axis=0)

N = X.shape[0]

# Normalize to 0–1 floats
X = X.astype(np.float32) / 255.0

# Flatten to (N, 784)
X = X.reshape(N, 28 * 28)
for i in range(784):
    print(X[i])
# Labels as uint8
y = y.astype(np.uint8)
# Export binary files
# X.tofile("mnist_images.bin")
# y.tofile("mnist_labels.bin")

# print(f"Export complete: {N} images")
