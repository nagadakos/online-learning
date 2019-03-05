import matplotlib.pyplot as plt
from matplotlib import markers
import numpy as np
import os
import sys
from pathlib import Path
from os.path import isdir, join, isfile
from os import listdir
import fnmatch
from itertools import cycle
from random import randint

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

def plot_regressor(filesPath='', title = '', xAxisNumbers = None, labels=[], inReps = [], plot = 'All', mode = 'Learning Curves'):
    
    ''' Description: This function will plot Learning or Prediciton curves, as supplied from either txt log files, or a list of
                     histories, or both. It returns a figure, containg all the curves; one curve for each history provided.

        Arguments:  filesPath(filePath): A file path to the folder containing the required log txt files.
                    title(String):       Title to the figure
                    xAxisNumbers(List):  A list of lalbel strings to be used as x axis annotations.
                    labels(List):        A list of strings to be used as curve labels.
                    inReps(List):        A list of model histories in the format train-loss MAE MAPE test-MAE MAPE loss.
                    plot (selector)      A string command  not yet offering functionality
                    mode(Selector):      A string command telling the function to plot Learning curves or simple prediction loss.
        Returns:    fig:     A figure object containg the plots.
    '''
    # Argument Handler
    # ----------------------
    # This section checks and sanitized input arguments.
    if not filesPath and  not inReps:
        print('No input log path or history lists are given to plot_regressor!!')
        print('Abort plotting.')
        return -1

    if not isinstance(filesPath, list):
        files = [filesPath]
    else:
        files = filesPath
    reps = []

    if filesPath:
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

    if inReps:
        for i,r in enumerate(inReps):
            # reps.append([[] for i in range(ridx.logSize)])
            reps.append(r)
    # print("Plots epochs: {}" .format(epochs))

    epochs = len(reps[0][0])
    if mode == 'Learning Curves':
        xLabel = 'Epoch'
    elif mode == 'Prediction History':
        xLabel = 'Task'

    if xAxisNumbers is None:
        epchs = np.arange(1, epochs+1)
    else:
        epchs = xAxisNumbers
    # ---|

    fig = plt.figure(figsize=(19.2,10.8))
    # fig = plt.figure(figsize=(13.68,9.80))
    plt.title(title)
    plt.ylabel('Loss')
    plt.xlabel(xLabel)
    # Set a color mat to use for random color generation. Each name is a different
    # gradient group of colors
    cmaps= ['Pastel1', 'Pastel2', 'Paired', 'Accent',
                     'Dark2', 'Set1', 'Set2', 'Set3',
                     'tab10', 'tab20', 'tab20b', 'tab20c']
    # Create an iterator for colors, for automated plots.
    cycol = cycle('bgrcmk')
    ext_list = []
    test_loss_list = []
    markerList = list(markers.MarkerStyle.markers.keys())[:-4] 
    for i, rep in enumerate(reps):
        # print(cmap(i))
        a = np.asarray(rep, dtype = np.float32)
        # WHen plotting multiple stuff in one command, keyword arguments go last and apply for all
        # plots.
        # If labels are given
        if not labels: 
            ext = os.path.split(files[i])[1].split('-') 
            ext = ' '.join(('lr', ext[0],'m',ext[1],'wD',ext[2]))
        else:
            ext = labels[i]
            print(ext)
        # Select color for the plot
        cSel = [randint(0, len(cmaps)-1), randint(0, len(cmaps)-1)]
        c1 = plt.get_cmap(cmaps[cSel[0]])
        # Solid is Train, dashed is test
        marker = markerList[randint(0, len(markerList))]
        if plot == 'All' or plot == 'Train':
            plt.plot(epchs, a[ridx.trainLoss], color = c1(i / float(len(reps))), linestyle =
                 '-', marker=marker, label = 'Train-'+ext)
        # plt.plot(epchs, a[ridx.testLoss],  (str(next(cycol))+markerList[rndIdx]+'--'), label = ext)
        if plot == 'All' or plot == 'Test': 
            linestyle = "-" if "no" in ext else "--"
            linestyle = ":" if "provided" in ext else linestyle 
            plt.plot(epchs, a[ridx.testLoss], color=  str(next(cycol)), linestyle = linestyle, marker=marker, label = 'Test-'+ext)
        plt.legend( loc='upper right')
        ext_list.append(ext)
        test_loss_list.append(a[ridx.testLoss][-1])

    best_index = np.argmin(np.array(test_loss_list))
    print("Best test loss is:", str(test_loss_list[best_index]))
    print("Best parameters are:", ext_list[best_index])
        # plt.close()
        # plt.draw()
        # plt.pause(15)

    return fig
# ----------------------------------------------------------------------------------------------------

def plot_prediction_loss(inReps, title, mode = 'trainHist'):
  print("hi") 
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
def plot_acc(rep1, rep2, epochs, title):
   
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

#--------------------------------------------------------------------------------------------------------
#
#--------------------------------------------------------------------------------------------------------
def main():
    # title = 'ANNGREEK Learning Curves Evaluation Solid: Train, Dashed: Test'+str(randint(0,20))
    figType = 'Solver_Performance_Curves' 
    # title = figType+'. Evaluation on Load: Task2. Model: ANNGREEK'
    title = figType+'. Online Prediction Scheme. Model: ANNGREEK constant lr:0.9, SGD-mom:0.5,wDecay:0.01, adaptive training stopage'
    # title = 'ANNGREEK Update Scheme Evaluation Plots'
    # filePath = "../../Applications/power_GEF_14/Logs/ANNGreek/Online/Experiment-02-10-2019-A-1-year/Prettyfied/Selective tests/DW-Cases"
    # experimentFolder = 'Experiment-02-10-2019-A-1-year'
    experimentFolder = 'Experiment-02-26-2019'
    targetExperiment = '/Online-vs-Offline'
    filePath = "../../Applications/power_GEF_14/Logs/ANNGreek/Online/"+experimentFolder+ targetExperiment

    f = get_files_from_path(filePath, "*.txt")
    # f = get_files_from_path(filePath, "*results.txt")
    # TODO: get these numbers automatically
    # print(f)
    files = []
    labels = []
    for i in f['files']:
        files.append(join(filePath, i)) 
        labels.append(i.rsplit('.',1)[0])
    # print(files)
    print(labels)
    print(len(files), len(labels))
        
    # NOTE: Change this number to match the training epochs
    xAxisNumbers = np.arange(2, 16)
    plot_regressor(filesPath = files, title=title, xAxisNumbers = xAxisNumbers, labels = labels, plot = 'Test', mode = 'Prediction History' )
    saveFile = "../../Applications/power_GEF_14/Plots/"+experimentFolder
    if not os.path.exists(saveFile):
            os.makedirs(saveFile)
    saveFile += '/'+figType+str(randint(0,150)) + ".png"
    plt.savefig(saveFile)
    plt.close()

  
if __name__ == '__main__':
    main()
