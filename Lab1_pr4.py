# In[4]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def Expected_value(n, samples):
    val0 = 0
    val1 = 0
    for sample in samples:
        val0 = val0 + sample[0]
        val1 = val1 + sample[1]
        
    val0 = np.divide(val0, n)
    val1 = np.divide(val1, n)
    return [val0, val1]

num_samples = 10000
cov_matrix = [[20, .8], [.8, 30]]
samples = np.random.multivariate_normal([-5, 5], cov_matrix, (num_samples))

# Calculating mean
mean = Expected_value(num_samples, samples)
print("mean = ", mean)

## Covariance matrix

# Basically transpose samples matrix to make for easier covariance calculations
sampleX = np.zeros(num_samples)
sampleY = np.zeros(num_samples)
for i in range(num_samples):
    sampleX[i] = samples[i][0]
    sampleY[i] = samples[i][1]    
samples_cov = np.array([sampleX, sampleY])

# Mean matrix for easier covariance calculations
dummy_mean_X = np.full(num_samples, mean[0])
dummy_mean_Y = np.full(num_samples, mean[1])
dummy_mean = np.array([dummy_mean_X, dummy_mean_Y])

cov = np.zeros([2, 2])
# Actual covariance calculations
for i in range(2):
    for j in range(2):        
        difference_i = np.subtract(samples_cov[i], dummy_mean[i])
        difference_j = np.subtract(samples_cov[j], dummy_mean[j])
        diff_mult = np.multiply(difference_i, difference_j)
        cov[i][j] = np.divide(np.sum(diff_mult), num_samples-1)
print("Covariance matrix: ")
print(cov)
