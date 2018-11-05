import matplotlib.pyplot as plt
import matplotlib.axes  as  ax
from matplotlib import cm
from labellines import labelLine, labelLines
import numpy as np
import os
import sys
from pathlib import Path
from os.path import isdir, join, isfile
from os import listdir
import fnmatch
from itertools import cycle

dir_path = os.path.dirname(os.path.realpath(__file__))

sys.path.insert(0, dir_path)

# import classification_idx as indexes
import regression_idx as ridx


epochs = 0

def get_files_from_path(targetPath, expression):

    # Find all folders that are not named Solution.
    d = [f for f in listdir(targetPath) if (isdir(join(targetPath, f)) and "Solution" not in f)]  
    # Find all file in target directory that match expression
    f = [f for f in listdir(targetPath) if (isfile(join(targetPath, f)) and fnmatch.fnmatch(f,
                                                                                            expression))]  
    # initialize a list with as many empty entries as the found folders.
    l = [[] for i in range(len(d))]
    # Create a dictionary to store the folders and whatever files they have
    contents = dict(zip(d,l))
    contents['files'] = f

    # Pupulate the dictionary with files that match the expression, for each folder.
    # This will consider all subdirectories of target directory and populate them with
    # files that match the expression.
    for folder, files in contents.items():
        stuff = sorted(Path(join(targetPath, folder)).glob(expression))
        for s in stuff:
            files.append(os.path.split(s)[1] )
        # print(folder, files)
    for files in contents['files']:
        stuff = sorted(Path(join(targetPath, files)).glob(expression))
    # print(contents)
    return contents

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
       RGB color; the keyword argument name must be a standard mpl colormap name.
    '''
    return plt.cm.get_cmap(name, n)

def plot_regressor(filesPath, args, title):

    if not isinstance(filesPath, list):
        files = [filesPath]
    else:
        files = filesPath
    reps = []
    for i,f in enumerate(files):
        reps.append([[] for i in range(ridx.logSize)])

        # print(i)
        # print("Size of reps list: {} {}".format(len(reps),len(reps[i])))
        with open(f, 'r') as p:
            # print("i is {}".format(i))
            for j,l in enumerate(p):
                # Ignore last character from line parser as it is just the '/n' char.
                report = l[:-2].split(' ')
                # print(report)
                reps[i][ridx.trainMAE].append(report[ridx.trainMAE])
                reps[i][ridx.trainMAPE].append(report[ridx.trainMAPE])
                reps[i][ridx.trainLoss].append(report[ridx.trainLoss])
                reps[i][ridx.testMAE].append(report[ridx.testMAE])
                reps[i][ridx.testMAPE].append(report[ridx.testMAPE])
                reps[i][ridx.testLoss].append(report[ridx.testLoss])

            epochs = len(reps[0][0])
    # print("Plots epochs: {}" .format(epochs))
    fig = plt.figure(figsize=(19.2,10.8))
    plt.title(title)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    cmap = get_cmap(2*len(reps))
    # Create an iterator for colors, for automated plots.
    cycol = cycle('bgrcmk')
    labels = []
    # for i, val in enumerate(cm.jet(10)):
        # print(val)
    for i, rep in enumerate(reps):
        # print(cmap(i))
        a = np.asarray(rep, dtype = np.float32)
        epchs = np.arange(1, epochs+1)
        # WHen plotting multiple stuff in one command, keyword arguments go last and apply for all
        # plots.
        ext = os.path.split(files[i])[1].split('-') 
        ext = ' '.join(('lr', ext[0],'m',ext[1],'wD',ext[2]))
        # Solid is Train, dashed is test
        plt.plot(epchs, a[ridx.trainLoss],  (str(next(cycol))+'-'), label = ext)
        plt.plot(epchs, a[ridx.testLoss],  (str(next(cycol))+'--'), label = ext)
        plt.legend( loc='lower right')
        # plt.close()
        # plt.draw()
        # plt.pause(15)
    # This will insert legend inline of curves
    # labelLines(plt.gca().get_lines(),align=False,fontsize=5)
    return fig

#************************************
#Function: Plot Accuracy
#Description:   This function will plot the accuracy curve of 2 statistics
 #              Reports, given they are in the format train acc, train loss
#               test acc, test loss.
#Arguments:     rep1:   list of stas report 1 
#               rep2:   list of stas report 2 
#               epochs: int. Number of epochs
#               title:  string, used to anotate the plot figure.
#***********************************
def plot_acc( rep1, rep2, epochs, title):
   
    a = np.asarray(rep1, dtype = np.float32)
    b = np.asarray(rep2, dtype = np.float32)
    ymin =np.asscalar(np.fmin(a[0][0], b[0][0]))
    print(ymin)
    epchs = np.arange(1, epochs+1)
    fig = plt.figure()
    plt.title(title)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    # plt.plot(epchs, a[0], 'r', label = 'Keras_train', epchs, b[0], 'b', 'PyTorch_train')
    plt.plot(epchs, a[0],  'r', epchs, b[0], 'b')
    plt.plot(epchs, a[2], 'm--', epchs, b[2], 'c--')
    labels = ['Keras_train', 'PyTorch_train', 'Keras_test', 'PyTorch_test']
    plt.legend( labels, loc='lower right')
    # plt.draw()
    # plt.pause(10)

# TODO: This Function will plot all reports on the same figure!
def plot_all_in_one(reps, epochs, title):

    print("Hellos")

def main():
    title = 'Multi-Linear Regression MSE Loss vs Epoch plot'
    # filePath = "../../Applications/power_GEF_14/Logs/log1.txt"
    filePath = "../../Applications/power_GEF_14/Logs/ANNGreek"

    f = get_files_from_path(filePath, "*log1.txt")
    # print(f)
    files = []
    for i in f['files']:
        files.append(join(filePath, i)) 
    # print(files)
    plot_regressor(files, 1, title)
    plt.savefig("../../Applications/power_GEF_14/Plots/test.png")
    plt.close()

  
if __name__ == '__main__':
    main()
