import numpy as np
from scipy import stats
data = np.random.normal(10, 1, 1000)
k2, p = stats.normaltest(data) #scipy.stats.normaltest() checking normality
alpha = 1e-3
if p < alpha:
    print("The data does not have a normal distribution")
else:
    print("The data has a normal distribution")
mean = np.mean(data)
print("The mean value of the data is", mean)
