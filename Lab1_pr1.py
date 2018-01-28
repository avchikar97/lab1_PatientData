# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

gaussian1 = np.random.normal(-10, 5, 1000)
gaussian2 = np.random.normal(10, 5, 1000)
gaussian3 = gaussian1 + gaussian2

count, bins, ignored = plt.hist(gaussian3, 50, normed=True)
plt.show()
