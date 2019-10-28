import pandas as pd

points = pd.read_csv('MDM Footscan Data\C. Measures\Default Measures.csv', header=0)
ball= points.iloc[:,2].values
heel= points.iloc[:,9].values

ratio= (ball/heel)
print(ratio)

