import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt


points = pd.read_csv('MDM Footscan Data\C. Measures\Default Measures.csv', header=0)
ball= points.iloc[:,2].values
instep= points.iloc[:,6].values
heel= points.iloc[:,9].values

ratio= (ball/heel)
print(ratio)

#plt.hist(ratio, bins=np.arange(min(ratio), max(ratio) +0.01, 0.01))
#plt.show

mean = np.mean(ratio)
print("mean =",(mean))

# skew = skew(ratio)
# print("skew =",(skew))

print("minimum =", min(ratio))
print("maximun =",max(ratio))

corr=pearsonr(heel,ball)
print(corr)

plt.scatter(heel, ball)
plt.show

