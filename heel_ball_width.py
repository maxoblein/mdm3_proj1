import pandas as pd
import numpy as np
import scipy.stats as sts
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#independency test 95% statistical significance
def independence(x,y):   
    r=sts.pearsonr(x,y)[0]
    print("r =",r)
    plt.figure()
    plt.scatter(x,y)
    Z=(np.sqrt(len(x)-3)/2)*np.log((1+r)/(1-r))
    if np.abs(Z)<=1.96:
        print("Data is independent to 95% statistical significance")
        return bool(1)
    else:
        print("Data is not independet to 95% statistical significance")
        return bool(0)

def linreg(x,y):
    x=np.array(x).reshape((-1,1))
    y=np.array(y)
    model = LinearRegression().fit(x, y)
    c=model.intercept_
    m=model.coef_
    print("y =",m[0],"x +",c)
    
def concave(heel,inst):
    out=[]
    for i in range(len(heel)):
        if heel[i]<inst[i]:
            out.append(0)
        else:
            out.append(1)
    return(out)

points = pd.read_csv('MDM Footscan Data\C. Measures\Default Measures.csv', header=0)
ball= points.iloc[:,2].values
instep= points.iloc[:,6].values
heel= points.iloc[:,9].values
ratio = heel/ball

independence(heel,ball)

independence(heel,ratio)
linreg(heel,ratio)

#print(concave(heel,instep))