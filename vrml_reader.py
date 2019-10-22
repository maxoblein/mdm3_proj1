import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import re
import sys
holder = []
with open(sys.argv[1], "r") as vrml:
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
    #print(holder)



holder_array = np.array(holder) #if you want numpy array
print(holder_array)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(holder_array[:,0],holder_array[:,1],holder_array[:,2],s = 0.1)
plt.show()
