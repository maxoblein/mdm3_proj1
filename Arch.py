import matplotlib.pyplot as plt
from vrml_reader import find_coords
from vrml_reader import Visualise
import numpy as np
import array as arr
import glob

    
def main():
    '''
    flatness = []

    for i in range(len(glob.glob1("","*.wrl"))):
        inp = glob.glob1("","*.wrl")[i]
        flat = FindArch(inp)        
        flatness.append(flat)
    #For the length of files ending in wrl use the data points in each file to find whether the foot is flat
    #create a list of points of flat(1) or not flat feet(0)
    '''
    '''
    flatness = np.vstack((flatness))
    '''
    #print(flatness)    
    
    FindArch('000006.wrl')
    
    
def FindArch(inp):
    
    data = find_coords(inp)
    
    #Visualise(data)
    #print(data)
    
    footprint = []
    #empty array 
        
    for i in range (len(data)):
       a = data[i, 1]
       if a < 0.0005  :
           b = np.array
           footprint.append(data[i])
    #For all data find where y values are less than ..., add those data points to the footprint array       
            
    footprint = np.vstack((footprint))
    #combine all arrays within foot print
    
    
    plt.scatter(footprint[:,0], footprint[:,2])
    #plots 2-D graph of foot print, using x values as x and y values as footprints z
    
    
    width_arch = []
    
    for i in range (len(footprint)):
        a = footprint[i,0]
        if 0.11 < a < 0.12:
            c = np.array
            width_arch.append(footprint[i])
    #for all data points in footprint, find x values between ... and ..., and create a new array.
    
    width_arch = np.vstack((width_arch))        
    #combines all arrays within width_arch        
    
    
    plt.scatter(width_arch[:,0], width_arch[:,2]) 
    #plots on top of the previous 2-D graph Width_arch values that will be used to calc the width
    
    
    width_arch_y = width_arch[:,2]
    minimum = np.amin(width_arch_y)
    maximum = np.amax(width_arch_y)
    #creates list of all y values and finds min and max of those values
    
    width = (maximum-minimum)   
    '''
    print('width of the Arch =', width)
    '''
    
    if width > 0.04:
        return 1
    else:
        return 0
    #defines whether a foot is flat or not, returning 1 or 0 
    
    
    
if __name__ == '__main__':
    argv = sys.argv[1:]
    main()
    
    