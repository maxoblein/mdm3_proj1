# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 10:08:08 2019

@author: James
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import numpy as np
import pandas as pd
import glob
import re
import sys


''' This is an array for the true values of the longest toe. 0 represents 
the big toe being the longest toe, 1 represents the index toe being the longest toe. '''

points = pd.read_csv("0-49.csv", header=None)

pred_longest = np.ones(len(points[2].values)-1)

observed_longest = np.ones(len(points[2].values)-1)
for i in range(1, len(points[2].values)):
    observed_longest[i-1] = np.int(points[2].values[i])


for t in range(50):
    inp = glob.glob1("","*.wrl")[t]

    foot_scan = inp

    def find_coords(filename):
        holder = []
        with open(filename, "r") as vrml:
            for i, line in enumerate(vrml):
                if i > 16:
                    for line in vrml:
                        if line[1] == ']':
                            vrml.close()
                            break
                        else:
                            line = line.strip(',\n')
                            #print(line)
                            new_line = line.split()
                            for j in range(len(new_line)):
                                new_line[j] = float(new_line[j])
                            holder.append(new_line)
    
                    break
        return np.array(holder)
        #print(holder)
        
    
    def formatting(ax):
        ax.set_xlabel('Width',fontsize = 18 )
        ax.set_ylabel('Length',fontsize = 18)
        ax.tick_params(labelsize=14)
        ax.set_title('Image of footscan ' + str(foot_scan),fontsize = 20)
        return ax
    
    ''' The following code is a 2D version of the visualiser function in order to
    produce a top down image of a footscan so that the toe line can be evaluated.'''  
    
    def Visualise2d(scan_array):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(scan_array[:,0],scan_array[:,1],s = 0.9)
        ax = formatting(ax)
        plt.show()
        
        
    def leftright(scan_array):
    
        leftside = np.ones(3)
    
        rightside = np.ones(3)
    
        for i in range(np.shape(scan_array)[0]):
    
            if scan_array[i,2] >=0.02:
    
                rightside = np.vstack((rightside,scan_array[i,:]))
    
            if scan_array[i,2] <= -0.02:
    
                leftside = np.vstack((leftside,scan_array[i,:]))
    
    
    
        leftside = np.delete(leftside,0,0)
    
        rightside = np.delete(rightside,0,0)
    
    
    
        maxleft = np.amax(leftside[:,0])
    
        maxright = np.amax(rightside[:,0])
    
        if maxright > maxleft:
    
            Side = 'L'
    
        else:
    
            Side = 'R'
        return Side
    
    
    '''A matrix containing the x and y coordinates for the footscan where 
    the first column is 'width' (x) and the second is 'length' (y).'''
    
    A = find_coords(foot_scan)
    A_2D = []
    A_2D = np.c_[A[:, 2], A[:, 0]]
    
    
    '''Whether a foot is left or right is imporant information and this 'foot' value is 
    used throughout.'''
    
    foot = leftright(A)
    
    
    '''A convex hull is made of the footscan using a built-in scipy function.'''
    
    C = ConvexHull(A_2D)
    
    
    '''The following image is a scatter plot of the footscan along with its convex hull which is represented
    by the outer black ring.'''
    
    Convex_array = []
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #ax.scatter(A_2D[:,0],A_2D[:,1],s = 0.9)
    #ax = formatting(ax)
    for simplex in C.simplices:
        Convex_array.append(A_2D[simplex])
        #plt.plot(A_2D[simplex, 0], A_2D[simplex, 1], 'k-')
        
    
    ''' A 2 dimensional array is made of the convex hull where each row is an x and y coordinate 
     representing a corner for the convex hull. '''
    
    
    Convex = np.zeros((np.shape(Convex_array)[0], np.shape(Convex_array)[1]))
    for i in range(len(Convex_array)):
        for j in range(2):
            Convex[i] = Convex_array[i][j]
            
            
    ''' Points that are less than 0.16 in the y-coorinate (along with their respective x values)
     are removed since they are not relevant. '''
    
    A_array = []
    for i in range(len(A_2D)):
        if A_2D[i, 1] > 0.16:
            A_array.append(A_2D[i]) 
            
    A_new = np.zeros((np.shape(A_array)[0], np.shape(A_array)[1]))
    for i in range(len(A_array)):
        A_new[i] = A_array[i]
        
    
    ''' A new convex hull array is made from the updated foot scan array. The method for 
    creating this convex hull and its array are identical to that shown previously. '''
    
    C_new = ConvexHull(A_new) 
    
    Convex_array = []
    #plt.plot(A_new[:,0], A_new[:,1], 'o')
    for simplex in C_new.simplices:
        Convex_array.append(A_new[simplex])
        #plt.plot(A_new[simplex, 0], A_new[simplex, 1], 'k-')
        

    ''' Points of the convex that are less than y = 0.18 are also removed since
    they are irrelevant. '''
    
    Toe_line_array = []
    for i in range (len(Convex)):
        if Convex[i][1] > 0.18:
            Toe_line_array.append(Convex[i])
            
    Toe_line = np.zeros((np.shape(Toe_line_array)[0], np.shape(Toe_line_array)[1]))
    for i in range(len(Toe_line_array)):
            Toe_line[i] = Toe_line_array[i] 
            

    '''The position of the big toe is found by looking for the maximum y-position in a certain range. 
    Since some feet have a longer index toe than big toe, we can't look for the max value
    across the entire range; an index toe is never less than x = -0.01 for a right foot or greater than
    x = 0.01 for a left foot, this is the range implemented. '''
    
    
    max_pos = 0
    max_value = 0
    
    
    if foot == 'L':
        for i in range (len(Toe_line)):
            j = Toe_line[i,1]
            if Toe_line[i][1] > max_value and Toe_line[i][0] > 0.01:
                max_value = Toe_line[i][1]
                max_pos = i
                
    if foot == 'R':
        for i in range (len(Toe_line)):
            j = Toe_line[i,1]
            if Toe_line[i][1] > max_value and Toe_line[i][0] < -0.01:
                max_value = Toe_line[i][1]
                max_pos = i
                
                
    ''' If the big toe is the maximum value across the entire range it is the longest toe, otherwise 
    it can be assumed that the index toe is the longest. '''
    
    
    if np.amax(Toe_line[:,1]) == max_value:
        pred_longest[t] = 0
    else:
        pred_longest[t] = 1
    

counter = 0
for i in range(len(observed_longest)):
    if pred_longest[i] == observed_longest[i]:
        counter = counter + 1


accuracy = np.float(counter) / np.float(len(observed_longest)) * 100



















