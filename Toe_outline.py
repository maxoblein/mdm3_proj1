import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from vrml_reader import find_coords
from scipy.spatial import ConvexHull, convex_hull_plot_2d

x = find_coords('000000.wrl')[:,(0,2)]

X = ConvexHull(x)
Line = [0,0]
for simplex in X.simplices:
    np.column_stack((Line, (x[simplex,0], x[simplex,1])))
#    plt.plot(x[simplex, 0], x[simplex, 1], 'k-')
print(Line)
#plt.plot(x[:,0], x[:,1])