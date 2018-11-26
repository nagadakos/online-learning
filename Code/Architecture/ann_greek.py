import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optm
import matplotlib.pyplot as plt
import numpy as np
import copy
import sys
import os 
from os.path import join
from random import shuffle

# Add this files directory location, so we can include the index definitions
dir_path = os.path.dirname(os.path.realpath(__file__))

# add this so we can import the trainer and tester modules
tools_path = os.path.join(dir_path, "../")
sys.path.insert(0, tools_path)

from Tools import trainer, plotter, utils
import Tools.regression_idx as ridx
import plot_idx as pidx


# End Of imports -----------------------------------------------------------------
# --------------------------------------------------------------------------------

sign = lambda x: ('+', '')[x < 0]
# CapWords naming convention for class names
# If there is a n acronym such as ANN, it is all uppercase
class ANNGreek(nn.Module):
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

    def __init__(self, inSize = 2, outSize = 1, loss = "Quantile", optim = "SGD", lr=0.1, momnt=0,
            wDecay=0, targetApp = "power_GEF_14", seed = 1): 
        super(ANNGreek, self).__init__() 

        # ===============================================================================
        # SECTION A
        # **********
        # NOTE: Change this appropriately for new Architectures! 
        #              
        # *******************************************************
        # Ceclare Name here. Used for saving and annotation.
        self.descr = "ANNGreek" 
        # ---|

        # Declare the layers here
        self.linear = nn.Linear(inSize, 24)  # 10 nodes are specified in the thesis.
        self.linear2 = nn.Linear(24, outSize)  # 10 nodes are specified in the thesis.
        # Set seed for random generator
        torch.manual_seed(seed)
        x = 0.5 
        nn.init.uniform_(self.linear.weight, -x, x)
        nn.init.uniform_(self.linear.bias, -x, x)
        nn.init.uniform_(self.linear2.weight, -x, x)
        nn.init.uniform_(self.linear2.bias, -x, x)
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
        self.defSavePrefix = '-'.join((str(self.lr),str(self.momnt), str(self.wDecay)))
        self.defPlotSaveTitle = '-'.join((self.descr, self.optim,"lr", str(self.lr),"momnt", str(self.momnt), str(self.wDecay), self.loss))    
        # End of init
        # ---------------------------------------------------------------------------------

    def forward(self, x): 

        # x = F.softmax(self.linear(x), dim=1) 
        # x = F.softmax(self.linear2(x), dim=0) 
        # x = F.relu(self.linear(x)) 
        # x = F.relu(self.linear2(x)) 
        x = F.elu(self.linear(x)) 
        x = F.elu(self.linear2(x)) 

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

    def save_history(self, filePath = None, rootFolder = '',  tarFolder = 'tempLogs', fileExt = '', savePredHist =
                    False, saveTrainHist = True, saveResults = False, results = None):

        if filePath is not None:
            saveFile = filePath
        else:
            rootFolder = self.defSavePrefix if rootFolder =='' else rootFolder
            saveFile = '/'.join(( self.defSavePath, 'Logs', self.descr, rootFolder, tarFolder))
            sep = '-'
            saveFile += '/' 
            # Create the Target Directory if does not exist.
            if not os.path.exists(saveFile):
                os.makedirs(saveFile)
            if saveResults == True:
                saveResFile = saveFile + sep.join((self.defSavePrefix, fileExt, ".txt"))
            saveFile += sep.join((self.defSavePrefix, fileExt, "log1.txt"))

        # Save training history or predHistory as required.
        if saveTrainHist == True:
            trainer.save_log(saveFile, self.history)
        if savePredHist == True :
            trainer.save_log(saveFile, self.predHistory)
        # Save Results if required
        if saveResults == True:
            if results is not None:
                try:
                    utils.save_tensor(results, filePath = saveResFile)
                except (AttributeError, TypeError):
                    raise AssertionError('Input Results variable should be Tensor.')
            else:
                print("No Results Tensor to save is given.")
        return saveFile

    def train(self, args, device, trainLoader, testLoader, optim, lossFunction =
              nn.MSELoss(),saveHistory = False, savePlot = False, modelLabel ='', saveRootFolder='',
             shuffleTrainLoaders = False):

        epochs = args[0]
        # Create deep copies of the arguments, might be required to the change. 
        trainerArgs = args.copy()
        testerArgs = args.copy()
        testerArgs[1] *= 4 

        # Make sure given train loader is a list
        if not isinstance(trainLoader, list):
            trainLoader = [trainLoader]
        
        # For each epoch train and then test on validation set.
        for e in range(epochs):
            # If random train order is required, randomize here. Keep in mind this is inplace.
            # It will affect the order of the input trainLoader list.
            if shuffleTrainLoaders == True:
                shuffle(trainLoader)
            trainerArgs[0] = e 
            testerArgs[0] = e 
            trainer.train_regressor(self, trainerArgs, device, trainLoader, optim, lossFunction = lossFunction)
            trainer.test_regressor(self, testerArgs, device, testLoader, lossFunction = lossFunction, trainMode= True)
        # If saving history and plots is required.
        fileExt = modelLabel + "-preTrain-for-"+str(epochs)+'-epchs'
        if saveHistory == True:
            self.save_history(tarFolder = 'PreTrain', fileExt = fileExt, rootFolder=saveRootFolder)
            print("Saving model {}-->id: {}".format(self.defPlotSaveTitle, hex(id(self))))

        # If no args for tarFolder are given plots go to the preTrain folder.
        # As: architect-0-optimName-lr-x-momnt-y-wD-z-LossName.png 
        if savePlot == True:
            self.plot(fileExt = fileExt, rootFolder=saveRootFolder)
            self.save_plots(titleExt = fileExt, saveRootFolder=saveRootFolder)
    
    # Testing and error reports are done here
    def predict(self, args, device, testLoader, lossFunction = nn.MSELoss(), saveResults = True,
                tarFolder = 'Predictions', fileExt = '', saveRootFolder = ''):
        taskLabel = args[2]
        print('Prediction mode  active')
        output, loss, lossMatrix = trainer.test_regressor(self, args, device, testLoader, lossFunction = lossFunction,
                               trainMode = False)

        # Only save the prediction history and the results, not the training history.
        if saveResults == True:
           saveRootFolder = saveRootFolder + '/' if saveRootFolder != '' else saveRootFolder
           self.save_history(tarFolder = saveRootFolder+tarFolder+'/PredHistoryLogs', fileExt = fileExt,
                             savePredHist = True, saveTrainHist = False) 
           self.save_history(tarFolder = saveRootFolder + tarFolder+'/PredResults', fileExt =
                             fileExt+taskLabel+'-lossMatrix', saveTrainHist = False, saveResults = True, results = lossMatrix) 
           self.save_history(tarFolder = saveRootFolder+tarFolder+'/PredResults', fileExt =
                             fileExt+taskLabel+'-predictions', saveTrainHist = False, saveResults = True, results = output) 


    def plot(self, filePath = None, logPath = None, rootFolder ='',tarFolder = 'PreTrain', fileExt = 'preTrain'):
        ''' Description: This function is a wrapper for the appropriate plot function
                         Found in the Tools package. It handles any architecture spec
                         cific details, that the general plot function does not, such
                         as: how many logs to read  and plot in the same graph.
            Arguments:   logPath  (string): Location of the log files to read.
        '''
        # Args is currently empty. Might have a use for some plotting arguments
        # In the future. Currently none are implemented.
        args = []
        rootFolder = self.defSavePrefix if rootFolder =='' else rootFolder
        if logPath is not None:
            readLog = logPath
        else:
            readLog = '/'.join(( self.defSavePath, 'Logs', self.descr, rootFolder, tarFolder+'/'))
            # readLog = dir_path + "/../../Applications/power_GEF_14/Logs/" + self.descr +'/'
            readLog += '-'.join((str(self.lr),str(self.momnt), str(self.wDecay), fileExt, "log1.txt"))
        # Form plot title and facilate plotting
        title = self.descr + " Learning Curve"
        self.plots[pidx.lrCurve] = plotter.plot_regressor(readLog, args,  title)

    # Save plots
    def save_plots(self, savePath = None, titleExt = None, saveRootFolder ='',tarFolder = 'PreTrain'):
        '''Description: This function saves all plots of model.
                        If no target path is given, the default is selected.
                        The default is the PreTrain folder of target architecture
                        and application.
        '''
        if savePath is None:
            savePath = '/'.join(( self.defSavePath, 'Plots', self.descr, saveRootFolder, tarFolder))
            # savePath = dir_path + "/../../Applications/power_GEF_14/Plots/" + self.descr
            # Create the Target Directory if does not exist.
            if not os.path.exists(savePath):
                os.makedirs(savePath)

        for i, f in enumerate(self.plots):
            if f is not None:
                fileExt = "/" + self.descr + "-" + str(i) + '-' + self.defPlotSaveTitle +'-'+ titleExt+ ".png"
            print("*****\nSaving figure: {} at {}****\n".format(self.descr, savePath + fileExt ))
            f.savefig(savePath + fileExt)

    def save(self, savePath = None, titleExt = '', tarFolder = ''):

        if savePath is None:

            savePath = '/'.join(( self.defSavePath, 'Models', self.descr, tarFolder))
            # Create the Target Directory if does not exist.
            if not os.path.exists(savePath):
                os.makedirs(savePath)
            fileExt = "/" + self.descr + "-" + self.defPlotSaveTitle + titleExt
            print("****\nSaving model: {}-->id: {} at {}\n****".format(self.descr, hex(id(self)), savePath + fileExt ))

        utils.save_model_dict(self, savePath+fileExt)

    def load(self, loadPath = None, titleExt = '', tarFolder = ''):

        if loadPath is None:
            loadPath = '/'.join(( self.defSavePath, 'Models', self.descr))
            fileExt = "/" + self.descr + "-" + self.defPlotSaveTitle + titleExt
            # Create the Target Directory if does not exist.
        if not os.path.exists(loadPath):
            print("Given path or model does not exists. Will try to load from defaualt location.")
        else:
            print("Loading saved model: {} to model {}@{}".format(loadPath, self.descr, hex(id(self))))

        utils.load_model_dict(self, loadPath)


    def print_out(self, mode='history'):

        if mode == 'History':
            print("Current stats of ANNSLF:")
            print("MAE:           {}" .format(self.history[ridx.trainMAE][-1]))
            print("MAPE:          {}" .format(self.history[ridx.trainMAPE][-1]))
            print("Training Loss: {}" .format(self.history[ridx.trainLoss][-1]))
            print("Test MAE:      {}" .format(self.history[ridx.testMAE][-1]))
            print("Test MAPE:     {}" .format(self.history[ridx.testMAE][-1]))
            print("Test Loss:     {}" .format(self.history[ridx.testLoss][-1]))
        if mode == 'params':
            print(list(self.parameters()))

    def create_copy(self, device, returnDataShape = 0):
           
        state_clone = copy.deepcopy(self.state_dict())
        model = ANNGreek(59, self.linear2.out_features, lr = self.lr, momnt=self.momnt,
                         wDecay=self.wDecay).to(device)
        model.load_state=dict(state_clone)
        reShapeDataTo = self.descr
        if returnDataShape == 1:
            return model, reShapeDataTo
        else:
            return model
