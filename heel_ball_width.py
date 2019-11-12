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
    print("Z =",Z)
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
    return [m[0],c]
    
def concave(heel,inst):
    out=[]
    for i in range(len(heel)):
        if heel[i]<inst[i]:
            out.append(0)
        else:
            out.append(1)
    return(out)

def formatting(ax):
    ax.set_xlabel('Ball width (mm)',fontsize = 18 )
    ax.set_ylabel('Heel width (mm)',fontsize = 18 )
    ax.tick_params(labelsize=14)
    ax.set_title('Relationship between Ball and Heel Width',fontsize = 20)
    return ax

points = pd.read_csv('MDM Footscan Data\C. Measures\Default Measures.csv', header=0)
ball= points.iloc[:,2].values
instep= points.iloc[:,6].values
heel= points.iloc[:,9].values
ratioh = heel/ball
ratiob = ball/heel

##indepedence testing
#independence(heel,ball)
#
#independence(heel,ratioh)

##first line fit
#coef=linreg(heel,ratioh)
##
#x=list(range(47,83))
#y=[]
#for item in x:
#    y.append(float(item)/(coef[0]*float(item)+coef[1]))
##print(y)
#
#plt.figure()
#ax=plt.subplot(1,1,1)
#ax.scatter(heel,ball,s=10)
#ax.plot(x,y,"r-")
#formatting(ax)
#plt.show()
##print(concave(heel,instep))

#second line fit
coef=linreg(ball,ratiob)
#
x=list(range(80,105))
y=[]
for item in x:
    y.append(float(item)/(coef[0]*float(item)+coef[1]))
#print(y)
    


plt.figure()
ax=plt.subplot(1,1,1)
ax.scatter(ball,heel,s=10)
ax.plot(x,y,"r-")
formatting(ax)
plt.show()
##print(concave(heel,instep))