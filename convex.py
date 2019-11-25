from vrml_reader import find_coords
from vrml_reader import leftright
#from vrml_reader import Visualise
import matplotlib.pyplot as plt
import numpy as np

def convex(data):
    
    #to chop off the ankle
    foot = []
    #empty array 
    for i in range (len(data)):
       a = data[i, 1]
       if a < 0.04  :
           foot.append(data[i])
    #For all data find where y values are less than ..., add those data points to the footprint array       
    foot = np.vstack((foot))
    #combine all arrays within foot print
    #plot to check
    #plt.scatter(foot[:,0], foot[:,2])
    
    #to separate heel
    heel = []
    #empty array 
    for i in range (len(foot)):
       a = foot[i, 0]
       if 0.0495 < a < 0.0505  :
           heel.append(foot[i])
    #For all data find where y values are less than ..., add those data points to the footprint array       
    heel = np.vstack((heel))
    #combine all arrays within foot print
    #plot to check
    #plt.scatter(heel[:,0], heel[:,2])
    
    #to separate instep
    instep = []
    #empty array
    for i in range (len(foot)):
       a = foot[i, 0]
       if 0.1395 < a < 0.1405 :
           instep.append(foot[i])
    #For all data find where y values are less than ..., add those data points to the footprint array       
    instep = np.vstack((instep))
    #combine all arrays within foot print
    #plot to check
    #plt.scatter(instep[:,0], instep[:,2])
    
    side=leftright(data)
    if side == "L":
        #find edge point of heel
        heel_edge=heel[np.argmin(heel[:,2])]
        #plt.scatter(heel_edge[0], heel_edge[2])
        #find edge point of instep
        instep_edge=instep[np.argmin(instep[:,2])]
        #plt.scatter(instep_edge[0], instep_edge[2])
    else:
        #find edge point of heel
        heel_edge=heel[np.argmax(heel[:,2])]
        #plt.scatter(heel_edge[0], heel_edge[2])
        #find edge point of instep
        instep_edge=instep[np.argmax(instep[:,2])]
        #plt.scatter(instep_edge[0], instep_edge[2])
        
    #check whether heel or instep were further out
    if abs(heel_edge[2])<abs(instep_edge[2]):
        convexity=str(1)
    else:
        convexity=str(0)
    return convexity
      
data = find_coords("MDM Footscan Data/VRML/000454.wrl")  
convex(data)