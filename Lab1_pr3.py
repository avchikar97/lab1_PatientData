# In[3]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def Expected_value(n, samples):
    ret = np.divide(np.sum(samples), n)
    return ret


num_samples = 25000
mu, sigma = 0, 5
samples = np.random.normal(mu, sigma, num_samples)
mean = Expected_value(num_samples, samples)

dummy_mean = np.full(num_samples, mean)
difference = np.subtract(samples, dummy_mean)
diff_sq = np.multiply(difference, difference)
variance = Expected_value(num_samples-1, diff_sq)

print("mean = ", mean)
print("sigma_square = ", variance)
