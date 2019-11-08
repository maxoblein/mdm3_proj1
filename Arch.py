import sys
import matplotlib.pyplot as plt
from vrml_reader import find_coords
from vrml_reader import Visualise
from check_all_scans import calculate_accuracy
from check_all_scans import get_true_labels
import numpy as np
import array as arr
import glob
import pandas as pd
import csv


    
def main():
    
    '''
    flatness = []

    for i in range(len(glob.glob1("","*.wrl"))):
        inp = glob.glob1("","*.wrl")[i]
        flat = FindArch(inp)        
        flatness.append(flat)
    #For the length of files ending in wrl use the data points in each file to find whether the foot is flat
    #create a list of points of flat(1) or not flat feet(0)
    
    
    #flatness = np.v;stack((flatness))
    for i in range(len(flatness)):
        flatness[i] = str(flatness[i])
    print(flatness)    
    '''
    
    FindArch('000036.wrl')
    
    # if you want to find the width of the arch of 1 particular data point, uncomment this and comment.
    
    
    #df = get_true_labels(r"C:\Users\user\Documents\Year 3\MDM3\Shoemaster\mdm3_proj1\flat_label.txt")
    #print(df)
    
    #accuracy, wronglist = calculate_accuracy(df, flatness)
   
    #print(accuracy)
    
    
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
    
    #fig = plt.figure()
    #ax = fig.subplots(111)
    
    plt.scatter(footprint[:,0], footprint[:,2])
    #ax = formatting(ax)
    #lt.show()
    
    
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
    
    plt.title('Footprint of 000004.wrl')
    plt.xlabel('Length')
    plt.ylabel('Width')

    
    width_arch_y = width_arch[:,2]
    minimum = np.amin(width_arch_y)
    maximum = np.amax(width_arch_y)
    #creates list of all y values and finds min and max of those values
    
    width = (maximum-minimum)   
    
    #print('width of the Arch =', width)
    
    #0.04
    if width >0.04:
        return 1
    else:
        return 0
    
    #defines whether a foot is flat or not, returning 1 or 0 
    
    #fig = plt.figure()
    #ax = fig.subplots(111)
    
    #ax.scatter(footprint[:,0], footprint[:,2])
    #ax.scatter(width_arch[:,0], width_arch[:,2]) 
    #ax = formatting(ax)
    #plots 2-D graph of foot print, using x values as x and y values as footprints z
    #plots on top of the previous 2-D graph Width_arch values that will be used to calc the width
    
    #plt.show()
    
    
if __name__ == '__main__':
    argv = sys.argv[1:]
    main()
    
    