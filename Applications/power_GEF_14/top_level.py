import sys as sys
import torch
import torchvision.datasets as tdata
import torchvision.transforms as tTrans
import os
from os.path import join
import matplotlib.pyplot as plt
import copy
import torch.nn as nn
import numpy as np
from random import randint
# WARNING:  This is relevant to the directory that CALLS this toplevel
# Module, rather than the position of this module.
dir_path   = os.path.dirname(os.path.realpath(__file__))
tools_path = os.path.join(dir_path, "../../Code/")
sys.path.insert(0, tools_path)


from  Solvers import sgd
from Datasets import GEF_Power
from Architecture import MLR, ann_forward, ann_greek
from Tools import trainer,plotter

# ================================================================================================================
# Start of Functions
# ================================================================================================================

def init_optim(modelParams, optimParams = dict(name="SGD", params=dict(lr=0.1,momnt=0.5,wDecay=0.9))):

    # Initialize the optimizer Template with the parameters of
    if optimParams["name"] == "SGD":
        optimTemplate = sgd.SGD(modelParams, weight_decay = optimParams["params"]["wDecay"][0],
                                lr=optimParams["params"]["lr"][0], momentum=optimParams["params"]["momnt"][0])
    else:
        optimTemplate = sgd.SGD(modelParams, weight_decay = optimParams["params"]["wDecay"][0],
                                lr=optimParams["params"]["lr"][0], momentum=optimParams["params"]["momnt"][0])

    return optimTemplate

# End of init_optim
#-----------------------------------------------------------------------------------------------------

def init(model = None, optimParams = dict(name="SGD", params=dict(lr=0.1,momnt=0.5,wDecay=0.9)), quantiles = [0.9], device = "cpu", trainDataRange = [0, 76799], testDataRange = [76800, 0], batchSize = 1000):
    ''' Description: This function handles the model creation with the chosen parameters and
                     the data loading with chosen batch size and train/test split.  

        Arguments: device: PyTorch identifier of model holder, CPU or GPU, if available .

                   train/test dataRange: Sample range, in raw data file indexing, of train 
                                         and test sets accordingly.

                   Batch size: Size of batch for data loading  
    '''
    #------------------------------------------------------------------------------
    # Architecture specifics ------------------------------------------------------

    # Model Declaration 
    # Generate as many models as there are parameters.
    models = []
    # Rename the parameter lists, loccaly, for ease of use.
    lrs     = optimParams['params']['lr']
    momnts  = optimParams['params']['momnt']
    wDecays = optimParams['params']['wDecay']
    # Compute the total num of models. Might be usefull.
    numOfModels = len(lrs)*len(momnts)*len(wDecays)
    outputSize = len(quantiles)
    idx = 0
    for l in range(len(lrs)):
        for m in range(len(momnts)):
            for w in range(len(wDecays)):
                if model == "ANNGreek":
                    models.append( ann_greek.ANNGreek(59, outputSize, optimParams['name'],
                                                      lr=lrs[l],momnt=momnts[m], wDecay=wDecays[w]).to(device))
                elif model == "MLRBIU":
                    models.append(ann_forward.ANNLFS().to(device))
                elif model == "MRLSimple": 
                    models.append( MLR.LinearRegression(25).to(device))
                elif "GLMLF" in model:
                    models.append( MLR.GLMLFB(model, outputSize).to(device))
                print("Creating Model: {}@{} at device: {}" .format(model,hex(id(models[idx])), device))
                idx += 1
    # ---|
    # Initialize the optimizer Template with the parameters of
    optimTemplate = init_optim(models[0].parameters(), optimParams)
    # -------------------------------------------------------------------------------
    # Data Loading Specifics --------------------------------------------------------

    # path has to be relative from the directory of the file OR terminal
    # that calls this specific top level, not the actual location of
    # this top level itself.
    #
    # Call this function to reshape raw data to match architecture specific inputs.
    # This function will reshape and save the data as: DataSet_reshaped_as_model.csv
    # delimitered by spaces.
    dataPath = None
    trainSet = GEF_Power.GefPower(dataPath, toShape = models[0].descr, transform =
                                  "normalize",dataRange= trainDataRange) 
    testSet = GEF_Power.GefPower(dataPath, toShape = models[0].descr, transform =
                                  "normalize",dataRange= testDataRange) 

    # Tell the Loader to bring back shuffled data, use 1 or more worker threads and pin-memory
    comArgs = {'shuffle': True,'num_workers': 2, 'pin_memory': True} if torch.cuda.is_available() else {}

    trainLoader = torch.utils.data.DataLoader(trainSet, batch_size = batchSize, **comArgs)
    testLoader = torch.utils.data.DataLoader(testSet, batch_size = batchSize, **comArgs)
    # Sanity prints
    print(len(testLoader.dataset))
    print(trainSet.__getitem__(0))

    return models, optimTemplate, trainLoader, testLoader 

# ------------------------------------------------------------------------------------------------------------------
# Main Function 
# ------------------------------------------------------------------------------------------------------------------
# parameter and model selection, here at section A. Section B has working logic, do not alter to run
# experiments.

def main():
    '''Description: This function is invoced then this top level is called.
                    It take the parsed arguments as input and will train,
                    teset and save the performance report of the endeavor.
                    The resulting plots are placed in the Plots folder.
                    The history is placed in the Logs folder.
    '''

    device  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    quantiles = [0.9]

    # ==========================================================================
    # SECTION A
    # **********
    # Start of Parameter definitions
    # All Parameter choices done within the section defined by the thick seperator
    # lines
    #****************************************************************************
    # Variable Definitions
    epochs = 200          # must be at least 2 for plot with labellines to work
    batchSize = 10000

    # Select Architecture here
    arch = "ANNGreek"
    # ---|

    # Loss Function Declaration and parameter definitions go here.
    quantiles = [0.01*i for i in range(1,100)]
    loss = trainer.QuantileLoss(quantiles)
    # loss = nn.MSELoss()
    # ---|

    # Optimizer Declaration and parameter definitions go here.
    gamma = [0.001, 0.003, 0.01, 0.3, 0.5]
    momnt = [0.1, 0.2, 0.3, 0.5, 0.7]
    wDecay= [0.1, 0.5]
    optimName = "SGD"
    totalModels = len(gamma) * len(momnt) * len(wDecay)
    optimParams = dict(name = optimName, params = dict(lr=gamma, momnt = momnt, wDecay=wDecay))
    # ---|

    # File, plot and log saving variables. Leaveto None, to save to default locations
    logSavePath  = None
    plogSavePath = None
    # ---|
    
    # **************************************************************************
    # End of parameter Definitions. Do not alter below this point.
    # ==========================================================================
    # SECTION B
    # **********
    # Start of working logic.
    # Pass this dictionary as arg input to the init function. The data ranges should be relevant
    # To the raw data files input. All offsetting etc is taken care of from the dataset code
    dataLoadArgs  = dict(model = arch, optimParams = optimParams, quantiles =quantiles, device = device, trainDataRange = [0, 76799], testDataRange = [76800, 0], batchSize = batchSize)

    # Get the models, optimizer, train and test data loader objects here
    # Models are initilized with the same weights. The number of models returned is
    # a function of the combinization of optimizer parameters lr*moment*weight decay.
    # Note: Do not change.
    modelsList, optimTemplate, trainLoader, testLoader = init(**dataLoadArgs)

    # Model Train Invocation and other handling here
    args = []
    args.append(epochs)
    args.append(batchSize)
    # List that holds all histories. 
    modelHistories = [modelsList[0].history for i in range(totalModels)]  

    # For each parameter evaluate a model. Each models keep track of its history, and the optimizer
    # params with which it was trained.  
    idx = 0
    for l in range(len(gamma)):
        for m in range(len(momnt)):
            for w in range(len(wDecay)):
                model = modelsList[idx]
                print("\nModel: {}@ {}. Lr: {} | Mom: {} | wDec: {}\n".format(idx,hex(id(model)), gamma[l], momnt[m],
                                                                      wDecay[w]))
                # NOTE: Deep copy and set parameters seems to not work. When used it probably
                # does not update the iptimizer to the new model's paramaters. Mystyriously the loss
                # is reduced phenomenally by this. Why?
                # optim = copy.deepcopy(optimTemplate)
                # optim.set_params(model.parameters(), lr=gamma[l], momentum=momnt[m], weight_decay=wDecay[w])
                optimParams = dict(name = optimName, params = dict(lr=[gamma[l]], momnt = [momnt[m]],
                                                                   wDecay=[wDecay[w]]))
                optim = init_optim(model.parameters(), optimParams)
                # Invoke training an Evaluation
                model.train(args,device, trainLoader, testLoader,optim, loss)
                # ---|

                # Report saving and printouts go here
                # print("Training history:")
                # print(model.history)
                model.save_history(logSavePath)
                modelHistories[idx] = model.history
                # plotArgs = []
                model.plot()
                lossDescr =  loss.descr if  isinstance(loss, trainer.QuantileLoss) else "MSE"
                # titleExt = optim.name + "-lr-" +  str(optim.lr) + "-momnt-" + str(optim.momnt)  +"-"+lossDescr
                titleExt = '-'.join((optim.name,"lr", str(optim.lr),"momnt", str(optim.momnt),
                                     str(optim.wDecay), lossDescr))

                print("Saving model {}{}-->id: {}".format(arch, titleExt, hex(id(model))))
                model.save_plots(plogSavePath, titleExt)
                idx += 1
    # ---|

    # Plot total evaluation plot
    filePath = dir_path + '/Logs/'+ arch
    f = plotter.get_files_from_path(filePath, "*log1.txt")
    print(f)
    files = []
    for i in f['files']:
        files.append(join(filePath, i)) 
    print("****\nPlotting Evaluation Curves...\n****")
    title = arch +' Learning Curves Evaluation\n Solid: Train, Dashed: Test'
    plotter.plot_regressor(files, 1, title)
    plt.savefig(dir_path + '/Plots/' + arch +'/eval-plot-'+str(randint(0,20))+'.png')
    plt.close()

#  End of main
#  -------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
