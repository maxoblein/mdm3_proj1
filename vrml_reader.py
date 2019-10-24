import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import re
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
    leftside =

if __name__ == '__main__':
    #on command line python vrml_reader.py 'Vrmlfile.wrl' option eg 'visualise', 'leftright'
    argv = sys.argv[1:]
    scan_array = find_coords(argv[0])
    if len(argv) >= 2:
        if argv[1] == 'visualise':
            Visualise(scan_array)

        if argv[1] == 'leftright':
            print('leftright')

    if len(argv) < 2:
        print(scan_array)
