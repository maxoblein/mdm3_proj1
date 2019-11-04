import pandas as pd
import numpy as np
import scipy.stats as sts
import matplotlib.pyplot as plt


points = pd.read_csv('MDM Footscan Data\C. Measures\Default Measures.csv', header=0)
ball= points.iloc[:,2].values
instep= points.iloc[:,6].values
heel= points.iloc[:,9].values

#ratio= (ball/heel)

#plt.hist(ratio, bins=np.arange(min(ratio), max(ratio) +0.01, 0.01))
#plt.show

#mean = np.mean(ratio)
#print("mean =",(mean))
#
#print("minimum =", min(ratio))
#print("maximun =",max(ratio))

r=sts.pearsonr(heel,ball)[0]

#independency test 95% statistical significance 
Z=(np.sqrt(len(heel)-3)/2)*np.log((1+r)/(1-r))
if np.abs(Z)<=1.96:
    print("Data is independent to 95% statistical significance")
else:
    print("Data is not independet to 95% statistical significance")

#plt.scatter(heel, ball)
#plt.show

