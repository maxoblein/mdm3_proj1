import sys
import os
import numpy as np
from vrml_reader import find_coords
from vrml_reader import leftright

if __name__ == '__main__':
    Label_list = []
    print(type(Label_list))
    directory = sys.argv[1]
    for filename in os.listdir(directory):
        scan_array = find_coords(os.path.join(directory,filename))
        Side = leftright(scan_array)
        Label_list.append(Side)


    print(Label_list)
