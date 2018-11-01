import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optm
import matplotlib.pyplot as plt
import numpy as np
import sys
import os 


# Add this files directory location, so we can include the index definitions
dir_path = os.path.dirname(os.path.realpath(__file__))

# add this so we can import the trainer and tester modules
tools_path = os.path.join(dir_path, "../")
sys.path.insert(0, tools_path)

from Tools import trainer, plotter
import Tools.regression_idx as ridx
import plot_idx as pidx


# End Of imports -----------------------------------------------------------------
# --------------------------------------------------------------------------------


sign = lambda x: ('+', '')[x < 0]
# CapWords naming convention for class names
# If there is a n acronym such as ANN, it is all uppercase
class GLMLFB(nn.Module):
    '''Description: General Linear Model Based Load Forecaster - Benchmark
                    This class implements a broad variaty of Linear model
                    for Load forecasting. These models are linear in weight 
                    space, not necesserailcy feature-space. Each one listed
                    here requires specially formed input. The appropriate raw
                    data transformation must be defined in the dataset class
                    that turns raw data input files, to PyTorch datasets,
                    found in the Datasets directory.

        Inputs:     (u int) Hourly Linear Trend: Each hour has its owh index, 
                            on the data set. Starting from a base date, each 
                            hour os enumerated. So f dataset starts at 
                            1-1-2001 00:00, hour 00:00 is 0, 01:00 is 1, 02:00 
                            is 2, etc.
                    (float) Temperature for that hour. If many are present in 
                            the dataset, start by using the average.


        Returns:    A single value prediction of the load.
    '''

    history = [[] for i in range(ridx.logSize)]
    def __init__(self, arch = "GLMLF-B7S", outputSize = 1, loss = nn.MSELoss()): 
        super(GLMLFB, self).__init__() 
        self.firstPass = 1
        if arch == "GLMLF-C2":
            inSize = 16 
        elif arch == "GLMLF-B7S":
            inSize = 47
        else:
            print("Unrecognized General Linear Model Load Forecaster Architecture")
            print("Initialize MRL layer with default size 25")
            inSize = 25
        self.linear = nn.Linear(inSize, outputSize)  # 10 nodes are specified in the thesis.
        self.loss = loss
        self.descr = arch
        # The list below holds any plotted figure of the model
        self.plots = [None] * pidx.plotSize

    def forward(self, x): 
        x = F.relu(self.linear(x)) 
        return x 

    def get_model_descr(self):
        """Creates a string description of a polynomial."""
        model = 'y = '
        modelSize =len(self.linear.weight.data.view(-1)) 
        for i, w in enumerate(self.linear.weight.data.view(-1)):
                model += '{}{:.2f} x{} '.format(sign(w), w, modelSize - i)
        model += '{:+.2f}'.format(self.linear.bias.data[0])
        return model   

    def shape_input(self, x, label):
        '''
            Shape unput data x to: 
                0-23 : Hourly load of last day
                24-27: Max and min temp of 2 weather station
                30-33: Bit encoding for day
        '''
         
        return x

    def save_history(self, filePath = None):

        if filePath is not None:
            saveFile = filePath
        else:
            saveFile = dir_path + "/../../Applications/power_GEF_14/Logs/" + self.descr
            # Create the Target Directory if does not exist.
            if not os.path.exists(saveFile):
                os.mkdir(saveFile)
            saveFile += "/log1.txt"

        trainer.save_log(saveFile, self.history)

    def train(self, args, device, trainLoader, testLoader, optim, lossFunction = nn.MSELoss()):

        epochs = args[0]
        
        trainerArgs = args.copy()
        testerArgs = args.copy()

        for e in range(epochs):
           trainerArgs[0] = e 
           testerArgs[0] = e 
           trainer.train_regressor(self, trainerArgs, device, trainLoader, optim, lossFunction)
           self.test(testerArgs, device, testLoader, lossFunction)
    
    # Testing and error reports are done here
    def test(self, args, device, testLoader, lossFunction = nn.MSELoss()):
        testArgs = args.copy()
        trainer.test_regressor(self, args, device, testLoader, lossFunction) 

    def plot(self, filePath = None, logPath = None):
        ''' Description: This function is a wrapper for the appropriate plot function
                         Found in the Tools package. It handles any architecture spec
                         cific details, that the general plot function does not, such
                         as: how many logs to read  and plot in the same graph.
            Arguments:   logPath  (string): Location of the log files to read.
        '''
        # Args is currently empty. Might have a use for some plotting arguments
        # In the future. Currently none are implemented.
        args = []
        if logPath is not None:
            readLog = logPath
        else:
            readLog = dir_path + "/../../Applications/power_GEF_14/Logs/"+self.descr+"/log1.txt" 
                # readLog = 
        # Form plot title and facilate plotting
        title = self.descr + " Learning Curve"
        self.plots[pidx.lrCurve] = plotter.plot_regressor(readLog, args,  title)

    # Save plots
    def save_plots(self, savePath = None, titleExt = None):
        '''Description: This function saves all plots of model.
                        If no target path is given, the default is selected.
        '''
        if savePath is None:
            savePath = dir_path + "/../../Applications/power_GEF_14/Plots/" + self.descr
            # Create the Target Directory if does not exist.
            if not os.path.exists(savePath):
                os.mkdir(savePath)

        for i, f in enumerate(self.plots):
            if f is not None:
                if titleExt is not None:
                    fileExt = "/" + self.descr + "-" + str(i) + "-" + titleExt + ".png"
                else:
                    fileExt = "/" + self.descr + "-" + str(i) + ".png"
            print("Saving figure: {} at {}".format(self.descr, savePath + fileExt ))
            f.savefig(savePath + fileExt)

    def report(self):

        print("Current stats of ANNSLF:")
        print("MAE:           {}" .format(self.history[ridx.trainMAE]))
        print("MAPE:          {}" .format(self.history[ridx.trainMAPE]))
        print("Training Loss: {}" .format(self.history[ridx.trainLoss]))
        print("Test MAE:      {}" .format(self.history[ridx.testMAE]))
        print("Test MAPE:     {}" .format(self.history[ridx.testMAE]))
        print("Test Loss:     {}" .format(self.history[ridx.testLoss]))


