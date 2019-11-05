import sys
import os
import numpy as np
from vrml_reader import find_coords
from vrml_reader import leftright
from arch import flatness

def calculate_accuracy(gt_labels, pred_labels):
    # write your code here (remember to return the accuracy at the end!)
    # gtlabels is the real labels, pred_labels are our predictions
    # calculate percentage of correctly classified, compare the same
    counter = 0
    incorrect_index_list = []
    for i in range(len(pred_labels)):
        if pred_labels[i] == gt_labels[i]:
            counter = counter + 1
        else:
            incorrect_index_list.append(i)

    accuracy = (counter / len(pred_labels)) * 100
    return accuracy, incorrect_index_list

def get_true_labels(filename):
    with open(filename,"r") as labels:
        CorrectLabels = []
        for line in labels:
            line = line.strip()
            label = str(line)
            CorrectLabels.append(label)
    return CorrectLabels

def get_approx_labels(directory):
    Label_list = []
    for filename in os.listdir(directory):
        scan_array = find_coords(os.path.join(directory,filename))
        Side = leftright(scan_array)
        Label_list.append(Side)
    return Label_list




if __name__ == '__main__':
    #on command line 1st argument is directory of scans second is file with true labels
    directory = sys.argv[1]
    CorrectLabels = get_true_labels(sys.argv[2])
    ApproxLabels = get_approx_labels(directory)
    accuracy, incorrect_index_list = calculate_accuracy(CorrectLabels,ApproxLabels)



    print(accuracy)
    print(incorrect_index_list)
