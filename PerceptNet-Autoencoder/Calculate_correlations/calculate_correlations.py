import tensorflow as tf
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from utils import *
from functions_correlation import calculate_correlation, plot_correlation_perceptnet_autoencoder

# Build the model: PerceptNetAutoEncoder or PerceptNetAutoEncoderOverComplete
perceptnet_autoencoder = PerceptNetAutoEncoderOverComplete(kernel_initializer = 'identity', gdn_kernel_size = 1)
perceptnet_autoencoder.compile(optimizer = 'adam', loss = PearsonCorrelation())
perceptnet_autoencoder.build(input_shape = (None, 384, 512, 3))

# Load pretrained weights for the model
weights_path = '../Over_complete/models/over_complete_123.h5'
perceptnet_autoencoder.load_weights(weights_path)           
print(perceptnet_autoencoder.summary())

# Calculate the correlations with TID-2013 (use "tid") or KADIK-10K (use "kadik") at every layer of the model
pearson, spearman = calculate_correlation(perceptnet_autoencoder, dataset = 'tid')
print(pearson)
print(spearman)

# Plot the correlations
plot_correlation_perceptnet_autoencoder(pearson, spearman, dataset = 'tid')