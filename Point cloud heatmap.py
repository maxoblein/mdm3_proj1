from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.models import model_from_json
from keras.models import load_model
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from vrml_reader import find_coords


#def point_cloud_viewer(point_data):
#    
#    data = []
#    for i in range(len(x)):
#        newpoint = x[i].split()[1:]
#        newpoint[2] = newpoint[2][:-2]
#        for j in range(len(newpoint)):
#            newpoint[j] = float(newpoint[j])
#            data.append(newpoint)
#    
#    datanp = np.array(data)
#
#    return datanp
# Function made Redundant by find_coords

def load_points_from_file(filename):
    points = pd.read_csv(filename, header=None, sep = 'delimiter', error_bad_lines = False, encoding = 'utf-8-sig')
    return points[0].values

def point_map(datanp, axes, size, cut1, cut2, cut3):
    axis1, axis2 = axes
    n,m = size
    cut = [cut1, cut2, cut3]
    A = np.zeros((n+1,m+1), dtype = int)
    for i in range(len(datanp)):
        if datanp[i, 0] >= cut1[0] and datanp[i, 0] <= cut1[1]:
            if datanp[i, 1] >= cut2[0] and datanp[i, 1] <= cut2[1]:
                if datanp[i, 2] >= cut3[0] and datanp[i, 2] <= cut3[1]:
                    n1 = (round(((datanp[i,axis1] - cut[axis1][0])*40), 1))
                    m1 = (round(((datanp[i,axis2] - cut[axis2][0])*40), 1))
                    A[int(n1*10),int(m1*10)] = A[int(n1*10),int(m1*10)] + 1
    A[A < 2] = 0
    return np.clip(A**3,0,20)

def Initialise_3d(inp, size, cut1, cut2, cut3, number, axes):
    #Returns a 3D array, consisiting of layers of 2D 'heatmap' representations of the points from the .wrl files inputted
    
    #inp = list of .wrl files from VRML folder
    #size = number of cells in 'heatmap' visualisation
    #cut1, cut2, cut3 = portion of the x,y,z points required, respectively (e.g. [0.05, 0.15] for points between 0.05 and 0.15)
    #number = range of files to be loaded (e.g. [50, 100])
    #axes = point of view of the foot (e.g. [0,2] will display the x and z axes - i.e. top down view)
    n,m = size
    X = np.zeros((n+1, m+1, int(number[1] - number[0])), dtype = int)
    for j in range(number[1] - number[0]):
        datanp = find_coords(inp[j+number[0]])
#        scan_array = find_coords(inp[j+number[0]])
#        leftside = np.ones(3)
#        rightside = np.ones(3)
#        for i in range(np.shape(scan_array)[0]):
#            if scan_array[i,2] >=0.02:
#                rightside = np.vstack((rightside,scan_array[i,:]))
#            if scan_array[i,2] <= -0.02:
#                leftside = np.vstack((leftside,scan_array[i,:]))
#            
#        leftside = np.delete(leftside,0,0)
#        rightside = np.delete(rightside,0,0)
#        
#        maxleft = np.amax(leftside[:,0])
#        maxright = np.amax(rightside[:,0])
#        if maxright > maxleft:
#            datanp[:,2] = datanp[:,2]*-1
        A = point_map(datanp, axes, [n, m], cut1, cut2, cut3)
        X[:,:,j] = X[:,:,j] + A[:,:]

    return X

def Initialise_3d_Swap(inp, size, cut1, cut2, cut3, number, axes):
    #Returns a 3D array, consisiting of layers of 2D 'heatmap' representations of the points from the .wrl files inputted
    
    #inp = list of .wrl files from VRML folder
    #size = number of cells in 'heatmap' visualisation
    #cut1, cut2, cut3 = portion of the x,y,z points required, respectively (e.g. [0.05, 0.15] for points between 0.05 and 0.15)
    #number = range of files to be loaded (e.g. [50, 100])
    #axes = point of view of the foot (e.g. [0,2] will display the x and z axes - i.e. top down view)
    n,m = size
    X = np.zeros((n+1, m+1, int(number[1] - number[0])), dtype = int)
    for j in range(number[1] - number[0]):
        datanp = find_coords(inp[j+number[0]])
        scan_array = find_coords(inp[j+number[0]])
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
            datanp[:,2] = datanp[:,2]*-1
        A = point_map(datanp, axes, [n, m], cut1, cut2, cut3)
        X[:,:,j] = X[:,:,j] + A[:,:]

    return X

def nn(inp, vec):

    X = Initialise_3d(inp, [40, 40], [0.2, 0.25], [0, 0.012], [-0.05, 0.05], [0, 300], [0,2])
    x = Initialise_3d(inp, [40, 40], [0.2, 0.25], [0, 0.012], [-0.05, 0.05], [501, 600], [0,2])
    #XS = Initialise_3d_Swap(inp, [40, 40], [0.15, 0.2], [0, 0.02], [-0.05, 0], [0, 1000], [0,2])
    #xS = Initialise_3d_Swap(inp, [40, 40], [0.15, 0.2], [0, 0.02], [-0.05, 0], [1001, 1289], [0,2])
    #XSA = Initialise_3d_Swap(inp, [40, 40], [0.1, 0.15], [0, 0.03], [-0.05, -0.01], [0, 1000], [1,0])
    #xSA = Initialise_3d_Swap(inp, [40, 40], [0.075, 0.16], [0, 0.03], [-0.05, -0.01], [1001, 1289], [1,0])
    
    #X = np.dstack((X,np.dstack((XS,XSA))))
    #x = np.dstack((x,np.dstack((xS,xSA))))
    #plt.imshow(X[:,:,2000], origin = 'upper')
    
    X = X.reshape(300, 41, 41, 1)
    x = x.reshape(99, 41, 41, 1)
    
    Y = to_categorical(vec[0:300])
    y = to_categorical(vec[501:600])
    
    loaded_model = Sequential()
    
    #loaded_model = load_model('model.h5')
    
    loaded_model.add(Conv2D(64, kernel_size = 3, activation = 'relu', input_shape = (41,41,1)))
    loaded_model.add(Conv2D(32, kernel_size = 3, activation = 'relu'))
    loaded_model.add(Flatten())
    loaded_model.add(Dense(2, activation = 'softmax'))
    ##64, 32 are the number of nodes; kernel_size is the size of the convolution matrix; relu activates the layers between nodes
    ##Dense is the layer type for the output layer, Flatten connects the convolution and dense layers; softmax makes the output sum to 1
    
    loaded_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    loaded_model.fit(X, Y, validation_data = (x,y), epochs = 5)
    
    #model_json = loaded_model.to_json()
    #with open("model.json", "w") as json_file:
    #    json_file.write(model_json)
    # serialize weights to HDF5
    #loaded_model.save_weights("model.h5")
    
    #json_file = open('model.json', 'r')
    #loaded_model_json = json_file.read()
    #json_file.close()
    #loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    #loaded_model.load_weights("model.h5")
    
    loaded_model.save('model.h5')
    
    print(loaded_model.predict(x[:10]))
    print(y[:10])
    
    
inp = os.listdir()
inp = inp[:-4]

LR = load_points_from_file('LR.csv')
integer_mapping = {x: i for i,x in enumerate(LR)}
vec = [integer_mapping[word] for word in LR]
vec = np.array(vec) - 1288


nn(inp, vec)