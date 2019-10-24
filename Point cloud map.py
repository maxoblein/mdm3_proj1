from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt


def point_cloud_viewer(point_data):
    
    data = []
    for i in range(len(x)):
        newpoint = x[i].split()[1:]
        newpoint[2] = newpoint[2][:-2]
        for j in range(len(newpoint)):
            newpoint[j] = float(newpoint[j])
            data.append(newpoint)
    
    datanp = np.array(data)

    return datanp

def load_points_from_file(filename):
    points = pd.read_csv(filename, header=None, sep = 'delimiter', error_bad_lines = False, encoding = 'utf-8-sig')
    return points[0].values

def point_map(datanp, axes, size, cut1, cut2):
    axis1, axis2 = axes
    n,m = size
    Z = datanp[:,[axis1,axis2]]
    A = np.zeros((n,m), dtype = int)
    for i in range(len(Z)):
        if Z[i, 0] >= cut1[0] and Z[i, 0] <= cut1[1]:
            if Z[i, 1] >= cut2[0] and Z[i, 1] <= cut2[1]:
                n1 = (round(((Z[i,0] - cut1[0])*30), 1))
                m1 = (round(((Z[i,1] - cut2[0])*30), 1))
                A[int(n1*10),int(m1*10)] = A[int(n1*10),int(m1*10)] + 1

    return A

inp = sys.argv[1]
x = load_points_from_file(inp)
datanp = point_cloud_viewer(x)
A = point_map(datanp, [0,2], [75, 30], [0, 0.25], [-0.05, 0.05])

plt.imshow(A)

#(X, Y), (x,y) = mnist.load_data()
#print(X[1])
#print(X[0].shape)

#plt.imshow(X[1])

#X = X.reshape(60000, 28, 28, 1)
#x = x.reshape(10000, 28, 28, 1)

#Y = to_categorical(Y)
#y = to_categorical(y)

#model = Sequential()

#model.add(Conv2D(64, kernel_size = 3, activation = 'relu', input_shape = (28,28,1)))
#model.add(Conv2D(32, kernel_size = 3, activation = 'relu'))
#model.add(Flatten())
#model.add(Dense(10, activation = 'softmax'))
#64, 32 are the number of nodes; kernel_size is the size of the convolution matrix; relu activates the layers between nodes
#Dense is the layer type for the output layer, Flatten connects the convolution and dense layers; softmax makes the output sum to 1

#model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

#model.fit(X, Y, validation_data = (x,y), epochs = 3)

#print(model.predict(x[:4]))
#print(y[:4])