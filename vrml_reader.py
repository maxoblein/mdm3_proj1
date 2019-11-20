import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import sys
import matplotlib.gridspec as gridspec



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
    fig = plt.figure(figsize = (7,5))
    ax = fig.add_subplot(111,projection = '3d')
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
def cut_off_top(scan_array):
    height_threshold = 0.025
    cut_off_array = np.ones(3)
    cut_off_index = np.argwhere(scan_array[:,1] < height_threshold)
    for i in cut_off_index:
        cut_off_array = np.vstack((cut_off_array,scan_array[i,:]))
    cut_off_array = np.delete(cut_off_array,0,0)
    return cut_off_array

def vectors_heel_ball_toe(cut_off_array,Side):
    array_2d = np.c_[cut_off_array[:, 2], cut_off_array[:, 0]]
    hull = ConvexHull(array_2d)
    Convex_list = []
    fig1 = plt.figure(figsize = (7,5))
    ax = fig1.add_subplot(111)
    ax.scatter(array_2d[:,1],array_2d[:,0])
    for simplex in hull.simplices:
        Convex_list.append(array_2d[simplex])
        ax.plot(array_2d[simplex, 1], array_2d[simplex, 0], 'k-')
    Convex_array = np.zeros((np.shape(Convex_list)[0], np.shape(Convex_list)[1]))
    for i in range(len(Convex_list)):
        for j in range(2):
            Convex_array[i] = Convex_list[i][j]

    plt.show(fig1)
    for i in range(np.shape(Convex_array)[0]):
        if Side == 'L':
            if i == np.argmax(Convex_array[:,0]):
                ball_point = Convex_array[i,:]
        if Side == 'R':
            if i == np.argmin(Convex_array[:,0]):
                ball_point = Convex_array[i,:]
    heel_list = []
    ind_list = []
    for j in np.argwhere(Convex_array[:,1] < 0.05):
        heel_list.append(Convex_array[j,0])
        ind_list.append(j)
    if Side == 'L':
        ind = np.argmax(np.array(heel_list))
    if Side == 'R':
        ind = np.argmin(np.array(heel_list))
    heel_point = Convex_array[ind_list[ind],:]
    print(heel_point[0])
    print(ball_point)





    fig2 = plt.figure(figsize = (7,5))
    ax2 = fig2.add_subplot(111)
    ax2.scatter(Convex_array[:,1],Convex_array[:,0])
    plt.show(fig2)




def is_ball_joint_inflamed(scan_array):
    Side = leftright(scan_array)
    cut_off_array = cut_off_top(scan_array)
    vectors_heel_ball_toe(cut_off_array,Side)
    #heel_ball = vector_heel_ball()
    #ball_toe = vector_ball_toe()
    #angle = np.acos(np.dot(heel_ball,ball_toe))
    #if angle > threshold:
    #    return 'True'
    #else:
    #    return 'False'


if __name__ == '__main__':
    #on command line python vrml_reader.py 'Vrmlfile.wrl' option eg 'visualise', 'leftright'
    argv = sys.argv[1:]
    scan_array = find_coords(argv[0])
    if len(argv) >= 2:
        if argv[1] == 'visualise':
            Visualise(scan_array)

        if argv[1] == 'leftright':
            print(leftright(scan_array))

        if argv[1] == 'inflamed':
            is_ball_joint_inflamed(scan_array)

    if len(argv) < 2:
        print(scan_array)
