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
    ax = formatting(ax)
    plt.show()

def formatting(ax):
    ax.set_xlabel('Length',fontsize = 18 )
    ax.set_ylabel('Height',fontsize = 18 )
    ax.set_zlabel('Width',fontsize = 18)
    ax.tick_params(labelsize=14)
    ax.set_title('Image of footscan 000000.wrl',fontsize = 20)
    return ax

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
            

    if len(argv) < 2:
        print(scan_array)
