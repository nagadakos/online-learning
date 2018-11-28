import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optm
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from itertools import chain
from os.path import join

# Add this files directory location, so we can include the index definitions
dir_path = os.path.dirname(os.path.realpath(__file__))

# add this so we can import the trainer and tester modules
tools_path = os.path.join(dir_path, "../")
sys.path.insert(0, tools_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from Tools import trainer, plotter, utils
import Tools.regression_idx as ridx
import plot_idx as pidx

# End Of imports -----------------------------------------------------------------
# --------------------------------------------------------------------------------

sign = lambda x: ('+', '')[x < 0]


# CapWords naming convention for class names
# If there is a n acronym such as ANN, it is all uppercase
class RNNsome(nn.Module):
    '''
        Simple single layer feedforwards network, for load preditction.
        It is found in Hond Tao's PhD dissertation as the entry level
        articificial neural netowork architecture.

        Inputs:     (u int) Hourly Linear Trend: Each hour has its owh index,
                            on the data set. Starting from a base date, each
                            hour os enumerated. So f dataset starts at
                            1-1-2001 00:00, hour 00:00 is 0, 01:00 is 1, 02:00
                            is 2, etc.
                    (float) Temperature for that hour. If many are present in
                            the dataset, start by using the average.


        Returns:    A single value prediction of the load.
    '''

    def __init__(self, inSize=2, outSize=1, loss="Quantile", optim="SGD", lr=0.1, momnt=0,
                 wDecay=0, targetApp="power_GEF_14"):
        super(RNNsome, self).__init__()

        # ===============================================================================
        # SECTION A
        # **********
        # NOTE: Change this appropriately for new Architectures!
        #
        # *******************************************************
        # Ceclare Name here. Used for saving and annotation.
        self.descr = "RNNsome"
        # ---|

        self.quantiles = [0.01*i for i in range(1,100)]
        self.num_quantiles = len(self.quantiles)
        self.hidden_layer = 64
        self.num_layer = 1
        # # Declare the layers here
        # self.lstm = nn.LSTM(
        #     inSize,
        #     self.hidden_layer,
        #     self.num_layer,
        #     batch_first=True) #(batch_size, time steps, features)
        #
        # final_layers = [nn.Linear(self.hidden_layer, 1) for _ in range(2)]
        # self.fc = nn.ModuleList(final_layers)
        self.dropout = 0.5
        self.in_shape = inSize
        self.out_shape = outSize
        self.build_model()
        self.init_weights()


        # self.fc = nn.Linear(self.hidden_layer, 99)
        # ---|

        # *******************************************************
        # Souldn't alter beneaththis point for testing!!
        # ===============================================================================


        self.firstPass = 1
        # The list below holds any plotted figure of the model
        self.plots = [None] * pidx.plotSize
        self.history = [[] for i in range(ridx.logSize)]
        self.predHistory = [[] for i in range(ridx.predLogSize)]
        # parameters here hold info for optim used to train the model. Used for annotation.
        self.loss = loss
        self.optim = optim
        self.lr = lr
        self.momnt = momnt
        self.wDecay = wDecay
        # ---|
        # Default File Saving parameter setting.
        self.targetApp = targetApp
        self.defSavePath = '/'.join((dir_path, '../../Applications', self.targetApp))
        self.defPlotSaveTitle = '-'.join(
            (self.descr, self.optim, "lr", str(self.lr), "momnt", str(self.momnt), str(self.wDecay), self.loss))
        # End of init
        # ---------------------------------------------------------------------------------

    def build_model(self):
        self.base_model = nn.Sequential(
            nn.Linear(self.in_shape, 64),
            nn.ReLU(),
            # nn.BatchNorm1d(64),
            nn.Dropout(self.dropout),
            nn.Linear(64, 64),
            nn.ReLU(),
            # nn.BatchNorm1d(64),
            nn.Dropout(self.dropout),
        )
        final_layers = [
            nn.Linear(64, 1) for _ in range(len(self.quantiles))
        ]
        self.final_layers = nn.ModuleList(final_layers)

    def init_weights(self):
        for m in chain(self.base_model, self.final_layers):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.unsqueeze(1)
        # h0 = torch.zeros(self.num_layer, x.size(0), self.hidden_layer).to(device)
        # c0 = torch.zeros(self.num_layer, x.size(0), self.hidden_layer).to(device)
        # out, _ = self.lstm(x, (h0, c0))
        # # out = self.fc(out[:, -1, :])
        # out = torch.cat([layer(out[:, -1, :]) for layer in self.fc], dim=1)
        # return out
        tmp_ = self.base_model(x)
        return torch.cat([layer(tmp_) for layer in self.final_layers], dim=1)

    def get_model_descr(self):
        """Creates a string description of a polynomial."""
        model = 'y = '
        # modelSize = len(self.linear.weight.data.view(-1))
        # for i, w in enumerate(self.linear.weight.data.view(-1)):
        #     model += '{}{:.2f} x{} '.format(sign(w), w, modelSize - i)
        # model += '{:+.2f}'.format(self.linear.bias.data[0])
        return model

    def shape_input(self, x, label):
        '''
            Shape unput data x to:
                0-23 : Hourly load of last day
                24-27: Max and min temp of 2 weather station
                30-33: Bit encoding for day
        '''

        return x

    def save_history(self, filePath=None, tarFolder='tempLogs', fileExt='', savePredHist=
    False, saveTrainHist=True, saveResults=False, results=None):

        if filePath is not None:
            saveFile = filePath
        else:
            saveFile = '/'.join((self.defSavePath, 'Logs', self.descr, tarFolder))
            # Create the Target Directory if does not exist.
            if not os.path.exists(saveFile):
                os.makedirs(saveFile)
            saveFile += '/'
            sep = '-'
            if saveResults == True:
                saveResFile = saveFile + sep.join((str(self.lr), str(self.momnt), str(self.wDecay),
                                                   fileExt, ".txt"))
            saveFile += sep.join((str(self.lr), str(self.momnt), str(self.wDecay), fileExt, "log1.txt"))

        # Save training history or predHistory as required.
        if saveTrainHist == True:
            trainer.save_log(saveFile, self.history)
        if savePredHist == True:
            trainer.save_log(saveFile, self.predHistory)
        # Save Results if required
        if saveResults == True:
            if results is not None:
                try:
                    utils.save_tensor(results, filePath=saveResFile)
                except (AttributeError, TypeError):
                    raise AssertionError('Input Results variable should be Tensor.')
            else:
                print("No Results Tensor to save is given.")
        return saveFile

    def train(self, args, device, trainLoader, testLoader, optim, lossFunction=
    nn.MSELoss(), saveHistory=False, savePlot=False):

        epochs = args[0]

        trainerArgs = args.copy()
        testerArgs = args.copy()
        testerArgs[1] *= 4

        for e in range(epochs):
            trainerArgs[0] = e
            testerArgs[0] = e
            trainer.train_regressor(self, trainerArgs, device, trainLoader, optim, lossFunction=lossFunction)
            trainer.test_regressor(self, testerArgs, device, testLoader, lossFunction=lossFunction, trainMode=True)
        # If saving history and plots is required.
        fileExt = "preTrain-for-" + str(epochs) + '-epchs'
        if saveHistory == True:
            self.save_history(tarFolder='PreTrain', fileExt=fileExt)
            print("Saving model {}-->id: {}".format(self.defPlotSaveTitle, hex(id(self))))

        # If no args for tarFolder are given plots go to the preTrain folder.
        # As: architect-0-optimName-lr-x-momnt-y-wD-z-LossName.png
        if savePlot == True:
            self.plot(fileExt=fileExt)
            self.save_plots()

    # Testing and error reports are done here
    def predict(self, args, device, testLoader, lossFunction=nn.MSELoss(), saveResults=True,
                tarFolder='Predictions', fileExt=''):
        taskLabel = args[2]
        print('Prediction mode  active')
        output, loss, lossMatrix = trainer.test_regressor(self, args, device, testLoader, lossFunction=lossFunction,
                                                          trainMode=False)

        # Only save the prediction history and the results, not the training history.
        if saveResults == True:
            self.save_history(tarFolder=tarFolder + '/PredHistoryLogs', fileExt=fileExt,
                              savePredHist=True, saveTrainHist=False)
            self.save_history(tarFolder=tarFolder + '/PredResults', fileExt=fileExt + taskLabel + '-lossMatrix',
                              saveTrainHist
                              =False, saveResults=True, results=lossMatrix)
            self.save_history(tarFolder=tarFolder + '/PredResults', fileExt=
            fileExt + taskLabel + '-predictions', saveTrainHist
                              =False, saveResults=True, results=output)

    def plot(self, filePath=None, logPath=None, tarFolder='PreTrain', fileExt='preTrain'):
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
            readLog = '/'.join((self.defSavePath, 'Logs', self.descr, tarFolder + '/'))
            # readLog = dir_path + "/../../Applications/power_GEF_14/Logs/" + self.descr +'/'
            readLog += '-'.join((str(self.lr), str(self.momnt), str(self.wDecay), fileExt, "log1.txt"))
        # Form plot title and facilate plotting
        title = self.descr + " Learning Curve"
        self.plots[pidx.lrCurve] = plotter.plot_regressor(readLog, args, title)

    # Save plots
    def save_plots(self, savePath=None, titleExt=None, tarFolder='PreTrain'):
        '''Description: This function saves all plots of model.
                        If no target path is given, the default is selected.
                        The default is the PreTrain folder of target architecture
                        and application.
        '''
        if savePath is None:
            savePath = '/'.join((self.defSavePath, 'Plots', self.descr, tarFolder))
            # savePath = dir_path + "/../../Applications/power_GEF_14/Plots/" + self.descr
            # Create the Target Directory if does not exist.
            if not os.path.exists(savePath):
                os.makedirs(savePath)

        for i, f in enumerate(self.plots):
            if f is not None:
                if titleExt is not None:
                    fileExt = "/" + self.descr + "-" + str(i) + "-" + titleExt + ".png"
                else:
                    fileExt = "/" + self.descr + "-" + str(i) + '-' + self.defPlotSaveTitle + ".png"
            print("*****\nSaving figure: {} at {}****\n".format(self.descr, savePath + fileExt))
            f.savefig(savePath + fileExt)

    def save(self, savePath=None, titleExt='', tarFolder=''):

        if savePath is None:

            savePath = '/'.join((self.defSavePath, 'Models', self.descr, tarFolder))
            # Create the Target Directory if does not exist.
            if not os.path.exists(savePath):
                os.makedirs(savePath)
            fileExt = "/" + self.descr + "-" + self.defPlotSaveTitle + titleExt
            print("****\nSaving model: {}-->id: {} at {}\n****".format(self.descr, hex(id(self)), savePath + fileExt))

        utils.save_model_dict(self, savePath + fileExt)

    def load(self, loadPath=None, titleExt='', tarFolder=''):

        if loadPath is None:
            loadPath = '/'.join((self.defSavePath, 'Models', self.descr))
            fileExt = "/" + self.descr + "-" + self.defPlotSaveTitle + titleExt
            # Create the Target Directory if does not exist.
        if not os.path.exists(loadPath):
            print("Given path or model does not exists. Will try to load from defaualt location.")
        else:
            print("Loading saved model: {} to model {}@{}".format(loadPath, self.descr, hex(id(self))))

        utils.load_model_dict(self, loadPath)

    def report(self):

        print("Current stats of ANNSLF:")
        print("MAE:           {}".format(self.history[ridx.trainMAE][-1]))
        print("MAPE:          {}".format(self.history[ridx.trainMAPE][-1]))
        print("Training Loss: {}".format(self.history[ridx.trainLoss][-1]))
        print("Test MAE:      {}".format(self.history[ridx.testMAE][-1]))
        print("Test MAPE:     {}".format(self.history[ridx.testMAE][-1]))
        print("Test Loss:     {}".format(self.history[ridx.testLoss][-1]))



