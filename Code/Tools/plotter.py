import matplotlib.pyplot as plt
import matplotlib.axes  as  ax
import numpy as np
import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))

sys.path.insert(0, dir_path)

# import classification_idx as indexes
import regression_idx as ridx


epochs = 0
# Sanity Print
# print(reps)

def plot_regressor(filePath, args, title):

    files = [filePath]
    reps = []
    for i,f in enumerate(files):
        reps.append([[] for i in range(ridx.logSize)])

        # print(i)
        # print("Size of reps list: {} {}".format(len(reps),len(reps[0])))
        with open(f, 'r') as p:
            # print("i is {}".format(i))
            for j,l in enumerate(p):
                # Ignore last character from line parser as it is just the '/n' char.
                report = l[:-1].split(' ')
                reps[i][ridx.trainMAE].append(report[ridx.trainMAE])
                reps[i][ridx.trainMAPE].append(report[ridx.trainMAPE])
                reps[i][ridx.trainLoss].append(report[ridx.trainLoss])
                reps[i][ridx.testMAE].append(report[ridx.testMAE])
                reps[i][ridx.testMAPE].append(report[ridx.testMAPE])
                reps[i][ridx.testLoss].append(report[ridx.testLoss])

            epochs = len(reps[0][0])
    print("Plots epochs: {}" .format(epochs))
    for i, rep in enumerate(reps):
        a = np.asarray(rep, dtype = np.float32)
        epchs = np.arange(1, epochs+1)
        fig = plt.figure()
        plt.title(title)
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        # plt.plot(epchs, a[0], 'r', label = 'Keras_train', epchs, b[0], 'b', 'PyTorch_train')
        plt.plot(epchs, a[ridx.trainLoss],  'r', epchs, a[ridx.testLoss], 'b')
        # plt.plot(epchs, a[2], 'm--', epchs, b[2], 'c--')
        labels = ['Train Loss', 'Test Loss']
        plt.legend( labels, loc='lower right')
        # plt.close()
        # plt.draw()
        # plt.pause(15)
        return fig

# def save_plots(savePath):

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
    filePath = "../../Applications/power_GEF_14/Logs/log1.txt"
    plot_regressor(filePath, 1, title)
    plt.savefig("../../Applications/power_GEF_14/Plots/MLR-25-epoch.png")
    plt.close()

  
if __name__ == '__main__':
    main()
