import numpy as np

def Right_to_Left(data):
    flipped_data = data
    leftside = np.ones(3)
    rightside = np.ones(3)
    for i in range(np.shape(data)[0]):
        if data[i,2] >=0.02:
            rightside = np.vstack((rightside,data[i,:]))
        if data[i,2] <= -0.02:
            leftside = np.vstack((leftside,data[i,:]))
    
    leftside = np.delete(leftside,0,0)
    rightside = np.delete(rightside,0,0)
    
    maxleft = np.amax(leftside[:,0])
    maxright = np.amax(rightside[:,0])
    if maxright < maxleft:
        flipped_data[:,2] = flipped_data[:,2]*-1
    
    return flipped_data