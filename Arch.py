import matplotlib.pyplot as plt
from vrml_reader import find_coords
from vrml_reader import Visualise
import numpy as np
import array as arr
    
inp = '000009.wrl'

data = find_coords(inp)

Visualise(data)
print(data)

size = len(data)
print(size)


footprint = []
#empty array 
    
for i in range (len(data)):
   a = data[i, 1]
   if a < 0.00001  :
       b = np.array
       footprint.append(data[i])
#For all data points where 'a' is less than some value, add those data points to the footprint array       
        
footprint = np.vstack((footprint))
#cobine all arrays within foot print
   
print(footprint)
print(len(footprint))
Visualise(footprint)   
    






#    if data([i,2]) > 0: 
#        np.delete(footprint, i, 0) 
#print(footprint)        
    
#footprint = find_coords(inp)

#Visualise(footprint)
    
    
    #if size(data(i,3) (>=) 0
     #  np.delete(data[i,:])
     
     #for i in range (len(data)):
 #   print(data[i])
#prints every part of array  
    
#footprint = data 
#a = footprint[0,2]
#if a > -1:
#    print('yes') 
        