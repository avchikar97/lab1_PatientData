# In[2]:

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import bernoulli

def Zn(n, draws):
    Zarray = np.zeros(draws)
    for i in range(draws):
        p = 0.5
        x0 = bernoulli.rvs(p, size=n)
        x1 = np.array(x0)
        np.place(x1, x0==0, -1)
        Zarray[i] = (1/n)*np.sum(x1)
    return Zarray

Z = Zn(5, 1000)
plt.hist(Z, 50)
plt.figure()
Z = Zn(30, 1000)
plt.hist(Z, 50)
plt.figure()
Z = Zn(250, 1000)
plt.hist(Z, 50)
plt.show()
