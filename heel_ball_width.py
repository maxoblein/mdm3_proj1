import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


points = pd.read_csv('MDM Footscan Data\C. Measures\Default Measures.csv', header=0)
ball= points.iloc[:,2].values
heel= points.iloc[:,9].values

ratio= (ball/heel)
print(ratio)

plt.hist(ratio)
plt.show