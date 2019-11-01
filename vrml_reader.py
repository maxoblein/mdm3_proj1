import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.spatial import ConvexHull
import sys

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

def Visualise(scan_array):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(scan_array[:,0],scan_array[:,1],scan_array[:,2],s = 0.9)
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

def slices(scan_array,side):
    front_array = np.ones(3)
    slice_range = np.linspace(scan_array[0,2],scan_array[-1,2],50)
    print(np.size(slice_range))
    for i in range(1,np.size(slice_range)):
        slice_array = np.ones(3)

        if side == 'L':
            slice_index = np.argwhere(np.logical_and(scan_array[:,2]<=slice_range[i-1],scan_array[:,2]>slice_range[i]))
        if side == 'R':
            slice_index = np.argwhere(np.logical_and(scan_array[:,2]>=slice_range[i-1],scan_array[:,2]<slice_range[i]))

        for j in slice_index:
            slice_array = np.vstack((slice_array,scan_array[j]))

        slice_array = np.delete(slice_array,0,0)

        front_point = slice_array[np.argmax(slice_array[:,0])]
        front_array = np.vstack((front_array,front_point))
    front_array = np.delete(front_array,0,0)
    print(front_array)
    return front_array

def inside_line_foot(scan_array):
    side = leftright(scan_array)
    inside_points = np.ones(3)
    slice_range = np.linspace(np.amin(scan_array[:,0]),np.amax(scan_array[:,0]))
    print(slice_range)
    for i in range(1,np.size(slice_range)):
        slice_array = np.ones(3)
        slice_index = np.argwhere(np.logical_and(scan_array[:,0]>=slice_range[i-1],scan_array[:,0]<slice_range[i],scan_array[:,1] < 0.01))
        print(slice_index)
        for j in slice_index:
            slice_array = np.vstack((slice_array,scan_array[j]))
        slice_array = np.delete(slice_array,0,0)
        if side == 'L':
            print(np.argmin(slice_array[:,2]))
            inside_points = np.vstack((inside_points,slice_array[np.argmin(slice_array[:,2])]))
        if side == 'R':
                inside_points = np.vstack((inside_points,slice_array[np.argmax(slice_array[:,2])]))
    inside_points = np.delete(inside_points,0,0)
    return inside_points

def Big_Toe_Isol8er(scan_array):
    tol = 1e-6
    side = leftright(scan_array)

    toe_index = np.argwhere(scan_array[:,0] > 0.15)
    toe_array = np.ones(3)
    for i in toe_index:
        toe_array = np.vstack((toe_array,scan_array[i]))
    toe_array = np.delete(toe_array,0,0)


    if side == 'L':
        #right side has big toe
        ind = np.lexsort((toe_array[:,0],toe_array[:,1],toe_array[:,2]))
        ind = np.flip(ind)
        sorted_array = toe_array[ind]
        front_array = slices(sorted_array,side)
        Visualise(scan_array)
        Visualise(front_array)



        return 0

    if side == 'R':
        #left side has big toe
        ind = np.lexsort((toe_array[:,0],toe_array[:,1],toe_array[:,2]))
        sorted_array = toe_array[ind]

        return 1


if __name__ == '__main__':
    #on command line python vrml_reader.py 'Vrmlfile.wrl' option eg 'visualise', 'leftright'
    argv = sys.argv[1:]
    scan_array = find_coords(argv[0])
    if len(argv) >= 2:
        if argv[1] == 'visualise':
            Visualise(scan_array)

        if argv[1] == 'leftright':
            print(leftright(scan_array))

        if argv[1] == 'toeline':
            #Big_Toe_Isol8er(scan_array)
            Big_Toe_Isol8er(scan_array)

    if len(argv) < 2:
        print(scan_array)
