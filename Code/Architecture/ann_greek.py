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
            wDecay=0, w= 1, targetApp = "power_GEF_14", seed = 1): 
        super(ANNGreek, self).__init__() 

        # ===============================================================================
        # SECTION A.0
        # ***********
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
        self.w = w
        # ---|
        # Default File Saving parameter setting.
        self.targetApp = targetApp 
        self.defSavePath = '/'.join((dir_path, '../../Applications', self.targetApp))
        self.defSavePrefix = '-'.join((str(self.lr),str(self.momnt), str(self.wDecay)))
        self.info ='-'.join((self.descr, self.defSavePrefix))
        self.defPlotSaveTitle = '-'.join((self.descr, self.optim,"lr", str(self.lr),"momnt", str(self.momnt), str(self.wDecay), self.loss))    
        # End of init
        # ---------------------------------------------------------------------------------

    def forward(self, x): 
        # =================================================================================
        # Section A.1
        # ***********
        # Define the architecture layout of the model
        # ******************************************************
        x = F.elu(self.linear(x)) 
        x = F.elu(self.linear2(x)) 

        return x 
        # ******************************************************
        # 
        # =================================================================================

    def get_model_descr(self):
        """Creates a string description of a polynomial."""
        model = 'y = '
        modelSize =len(self.linear.weight.data.view(-1)) 
        for i, w in enumerate(self.linear.weight.data.view(-1)):
                model += '{}{:.2f} x{} '.format(sign(w), w, modelSize - i)
        model += '{:+.2f}'.format(self.linear.bias.data[0])
        return model   

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
             shuffleTrainLoaders = False, adaptDurationTrain = False, epsilon = 0.1):

        epochs = args[0]
        # Create deep copies of the arguments, might be required to the change. 
        trainerArgs = args.copy()
        testerArgs = args.copy()
        testerArgs[1] *= 4 
        l_t = 0
        l_t_1 = 0
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
            p,l_t,lM = trainer.test_regressor(self, testerArgs, device, testLoader, lossFunction = lossFunction, trainMode= True)
            # If adaptive duration for training is enabled, then break training process if progress
            # in loss is less than epsilon. Should be expanded in average over past few epochs.
            if adaptDurationTrain == True:
                if trainer.dynamic_conv_check(self.history, args=dict(window= 10, percent_change=0.03, counter =e)):
                    break
                else:
                    print("CONTINUE")
        # If saving history and plots is required.
        fileExt = modelLabel + "-for-"+str(epochs)+'-epchs'
        if saveHistory == True:
            self.save_history(tarFolder = 'PreTrain', fileExt = fileExt, rootFolder=saveRootFolder)
            print("Saving model {}-->id: {}".format(self.defPlotSaveTitle, hex(id(self))))

        # If no args for tarFolder are given plots go to the preTrain folder.
        # As: architect-0-optimName-lr-x-momnt-y-wD-z-LossName.png 
        if savePlot == True:
            self.plot(fileExt = fileExt, rootFolder=saveRootFolder)
            self.save_plots(titleExt = fileExt, saveRootFolder=saveRootFolder)
   
#----------------------------------------------------------------------------------------------------
#
#----------------------------------------------------------------------------------------------------
    # Testing and error reports are done here
    def predict(self, args, device, testLoader, lossFunction = nn.MSELoss(), saveResults = True,
                tarFolder = 'Predictions', fileExt = '', saveRootFolder = '', modelLabel = ''):
        taskLabel = args[2]
        print('Prediction mode  active')
        output, loss, lossMatrix = trainer.test_regressor(self, args, device, testLoader, lossFunction = lossFunction,
                               trainMode = False)

        # If saving history and plots is required.
        fileExt = modelLabel 
 
        saveRootFolder = saveRootFolder + '/' if saveRootFolder != '' else saveRootFolder
        # Save prediction history losses
        self.save_history(tarFolder = tarFolder+'/PredHistoryLogs', rootFolder = saveRootFolder,fileExt = fileExt,
                             savePredHist = True, saveTrainHist = False) 
        # Only save the prediction results, not the training history.
        if saveResults == True:

            # Save prediction loss matrices
            self.save_history(tarFolder = tarFolder+'/PredResults', rootFolder = saveRootFolder, fileExt =
                             fileExt+taskLabel+'-lossMatrix', saveTrainHist = False, saveResults = True, results = lossMatrix) 
            # Save prediction actual results.
            self.save_history(tarFolder = tarFolder+'/PredResults', rootFolder =
                             saveRootFolder, fileExt =
                             fileExt+taskLabel+'-predictions', saveTrainHist = False, saveResults = True, results = output) 


    def plot(self, mode = 'Learning Curves', source = 'Logs', filePath = None, logPath = None, rootFolder ='',tarFolder = 'PreTrain', fileExt = 'preTrain'):
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
        if source == 'Logs':
            if logPath is not None:
                readLog = logPath
            else:
                readLog = '/'.join(( self.defSavePath, 'Logs', self.descr, rootFolder, tarFolder+'/'))
                # readLog = dir_path + "/../../Applications/power_GEF_14/Logs/" + self.descr +'/'
                readLog += '-'.join((str(self.lr),str(self.momnt), str(self.wDecay), fileExt, "log1.txt"))
            # Form plot title and facilate plotting
            title = self.descr + ' '+mode
            self.plots[pidx.lrCurve] = plotter.plot_regressor(filesPath = readLog, title = title, mode = mode)
        elif source == 'History':
            title = self.descr + mode
            self.plots[pidx.lrCurve] = plotter.plot_regressor(inReps = self.history,
                                                                        title=title, mode=mode)
            self.plots[pidx.predCurve] = plotter.plot_regressor(inReps =self.predHistory,
                                                                          title=title,
                                                                          mode = 'Prediction History')

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
        ''' Description: This function will create a ne deep copy of this model.
            
            Arguments:  device:          Where the model is instantiated, GPu or CPU
                        returnDataShape: Flag of whether a return string is required for raw data
                                         reshaping 
                        
            returns:    model: A model object, with all the parameters of the initial one, deep-copied. 
        '''
                
        state_clone = copy.deepcopy(self.state_dict())
        model = ANNGreek(59, self.linear2.out_features, lr = self.lr, momnt=self.momnt,
                         wDecay=self.wDecay).to(device)
        model.load_state=dict(state_clone)
        reShapeDataTo = self.descr
        if returnDataShape == 1:
            return model, reShapeDataTo
        else:
            return model
