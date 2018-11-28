import sys as sys
import torch
import torchvision.datasets as tdata
import torchvision.transforms as tTrans
import os
from os.path import join
import matplotlib.pyplot as plt
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
from Tools import trainer,plotter,utils

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
def load_data(preTrainYears, dataPath = None, batchSize = 1000,tasks = 'All', loadFromEnd = False, reshapeDataTo = None):

    # Task 1 file offset.
    if loadFromEnd == True:
        trainDataRange = [85443 - int(preTrainYears * 365.25)*24, 0]
        testDataRange = trainDataRange
    else:
        trainDataRange = [0, 35064 + int(preTrainYears * 365.25)*24] #if trainingScheme['update'] != 'Benchmark' else [0, 90000]
        testDataRange = [trainDataRange[1], 0]
    print(trainDataRange)

    trainSet = GEF_Power.GefPower(dataPath, toShape = reshapeDataTo, transform =
                                  "normalize",dataRange= trainDataRange) 
    valSet = GEF_Power.GefPower(dataPath, toShape = reshapeDataTo, transform =
                                  "normalize",dataRange= testDataRange) 

    # Tell the Loader to bring back shuffled data, use 1 or more worker threads and pin-memory
    comArgs = {'shuffle': True,'num_workers': 2, 'pin_memory': True} if torch.cuda.is_available() else {}

    trainLoader = torch.utils.data.DataLoader(trainSet, batch_size = batchSize, **comArgs)
    valLoader = torch.utils.data.DataLoader(valSet, batch_size = batchSize, **comArgs)

        # If required return the task loaders for the online schemes.
    filesNum = tasks if tasks != "All" else [1*i for i in range(2,16)]
    taskLoaders = []
    for i, t in enumerate(filesNum):
        testSet = GEF_Power.GefPower(dataPath, task ='Task ' +str(t), toShape = reshapeDataTo, transform = "normalize",dataRange= testDataRange) 
        testLoader = torch.utils.data.DataLoader(testSet, batch_size = batchSize, **comArgs)
        taskLoaders.append(testLoader)

    return trainLoader, valLoader, taskLoaders, filesNum

# End of load_data
#-----------------------------------------------------------------------------------------------------

# def init_online_data(model = None, tasks = "All", device = "cpu", trainDataRange = [0, 76799], testDataRange = [76800, 0], batchSize = 1000):


    # comArgs = {'shuffle': True,'num_workers': 2, 'pin_memory': True} if torch.cuda.is_available() else {}
    # filesNum = tasks if tasks is not "ALL" else [1 for i in range(2,15)]
    # taskLoaders = []
    # dataPath = None
    # for i, t in enumerate(filesNum):
        # testSet = GEF_Power.GefPower(dataPath, task ='Task ' +t, toShape = model, transform = "normalize",dataRange= testDataRange) 
        # testLoader = torch.utils.data.DataLoader(testSet, batch_size = batchSize, **comArgs)
        # taskLoaders.append(testLoader)
    # return taskLoaders

def init(model = None, tasks = "All", optimParams = dict(name="SGD", params=dict(lr=0.1,momnt=0.5,wDecay=0.9)),
         quantiles = [0.9], device = "cpu", trainingScheme =dict(preTrainOn = ['5 Years'] , update ='Benchmark') , batchSize = 1000):
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
                    reshapeDataTo = model
                    models.append( ann_greek.ANNGreek(59, outputSize, optimParams['name'],
                                                      lr=lrs[l],momnt=momnts[m], wDecay=wDecays[w]).to(device))
                elif model == "MLRBIU":
                    models.append(ann_forward.ANNLFS().to(device))
                elif model == "MRLSimple": 
                    models.append( MLR.LinearRegression(25).to(device))
                elif "GLMLF" in model:
                    reshapeDataTo = model
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
    dataPath = None
    # How many years toallocate for pretraining. use loadFromend, to load data from the end towards
    # the start. This is usefull if you want to load only the later years and thus limit your
    # available training data.
    loadFromEnd = True if trainingScheme['update'] == 'Offline' else False
    preTrainYears = int(trainingScheme['preTrainOn'][0][0]) if trainingScheme['update'] != 'Benchmark' else 5.75
    # Call this function to reshape raw data to match architecture specific inputs.
    # This function will reshape and save the data as: DataSet_reshaped_as_model.csv
    # delimitered by spaces.
    #
    # Will create data loaders for: pretrain, validation, prediction tasks and a list of pred tasks
    # numbers
    trainLoader, valLoader, taskLoaders, filesNum = load_data(preTrainYears, loadFromEnd =
                                                              loadFromEnd, batchSize= batchSize, reshapeDataTo = reshapeDataTo)

    # Sanity prints
    print(len(taskLoaders[0].dataset))
    # print(testSet.__getitem__(0))
    
    # ------------------------------------------------------------------------------    -
    # SCHEDULER
    # *********
    # Generate a schedule of training and testing schemes.
    schedule = dict( trainOn = [], testOn = [], predOn = [], labels =[], testLabels =[], predLabels
                   = [])
    # Benchmark creates a benchmark case for online training.
    # It uses naively, all available data up to a point to train, and 
    # tests on the rest. I.e train on Task1, 2 ,3 and test on task 4: end.
    # The trainset start from Task 1, Task 1 + val, Task 1 + val + Task 2...
    # The test set Val + Task loaders, Taskloaders, TaskLoaders[2:end] ...
    if trainingScheme['update'] == 'Benchmark':
        schedule['trainOn'].append([trainLoader])
        schedule['testOn'].append([taskLoaders[0]])
        schedule['labels'].append(['Task 1'])
        schedule['testLabels'].append(['Task ' + str(filesNum[0])])
        for i, t in enumerate(filesNum[:-1]):
            print(i, t)
            prevTrain = schedule['trainOn'][i].copy()
            prevLabels  = schedule['labels'][i].copy()
            prevTrain.append(taskLoaders[i])
            prevLabels.append('Task '+str(i+2))
            schedule['trainOn'].append(prevTrain)
            schedule['testOn'].append([taskLoaders[i+1]])
            schedule['labels'].append(prevLabels)
            schedule['testLabels'].append(['Task ' + str(filesNum[i+1])])
    elif trainingScheme['update'] == 'ParamEvaluation':
        schedule['trainOn'].append([trainLoader])
        schedule['testOn'].append([valLoader])
        schedule['labels'].append(['Task 1'])
        schedule['testLabels'].append(['EvalSet ' + str(5-preTrainYears)])
    elif trainingScheme['update'] == 'Offline':
        schedule['trainOn'].append([trainLoader])
        schedule['testOn'].append([valLoader])
        trainOrder = 'first' if loadFromEnd == False else 'last'
        schedule['labels'].append(['-'.join(('Task 1-OffLine-trainedOn',trainOrder, str(preTrainYears)))])
        schedule['testLabels'].append(['EvalSet ' + str(5-preTrainYears)])
        for i, t in enumerate(filesNum):
            print(i, t)
            schedule['predOn'].append([taskLoaders[i]])
            schedule['predLabels'].append(['Pred on Task ' + str(filesNum[i])])
    elif trainingScheme['update'] == 'Default':
        schedule['trainOn'].append([trainLoader])
        schedule['testOn'].append([valLoader])
        schedule['labels'].append(['Task 1-Default'])
        schedule['testLabels'].append(['EvalSet ' + str(5-preTrainYears)])
        for i, t in enumerate(filesNum):
            print(i, t)
            schedule['predOn'].append([taskLoaders[i]])
            schedule['predLabels'].append(['Pred on Task ' + str(filesNum[i])])
    # print(schedule)
    # print(schedule['testOn'])
    # print(schedule['testLabels'])
    # ---|
    return models, optimTemplate, trainLoader, valLoader, taskLoaders, schedule

# ------------------------------------------------------------------------------------------------------------------
# Main Function 
# ------------------------------------------------------------------------------------------------------------------
# parameter and model selection, here at section A. Section B has working logic, do not alter to run
# experiments.

def main():
    '''DESCRiption: This function is invoced then this top level is called.
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
    epochs = 15         # must be at least 2 for plot with labellines to work
    batchSize = 1000

    # Select Architecture here
    arch = "ANNGreek"
    # ---|

    # Loss Function Declaration and parameter definitions go here.
    quantiles = [0.01*i for i in range(1,100)]
    # quantiles = [0.1,0.5,0.7, 0.9]
    loss = utils.QuantileLoss(quantiles)
    # loss = nn.MSELoss()
    # ---|

    # Optimizer Declaration and parameter definitions go here.
    gamma = [0.5]#, 0.9]
    momnt = [0.1]#, 0.5]
    wDecay= [0.01]
    optimName = "SGD"
    totalModels = len(gamma) * len(momnt) * len(wDecay)
    optimParams = dict(name = optimName, params = dict(lr=gamma, momnt = momnt, wDecay=wDecay))
    # ---|

    # Task loading and online learning params go here.
    tasks = 'All' # Use this to load all Tasks 
    availSchemes = ['Benchmark', 'ParamEvaluation', 'Default', 'Offline']
    scheme = availSchemes[3] # Benchmark, ParamEvaluation
    trainingScheme= dict(preTrainOn = ["1 Years"], update = scheme)   # set has 5.75 years. Update: monthly, weekly,
                                                                         # benchmark
    # tasks = [2, 4, 6, 8] # Use this to provide a list of the required subset!
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
    # The tasks argument controls which Task folders are going to be loaded for use. Give a list
    # containing the numbers for the task you want i.e [2, 4,5,6,7,12]. If you with to load all
    # Simple set tasks  "All", which is also the default value.
    dataLoadArgs  = dict(model = arch, tasks = tasks, optimParams = optimParams, quantiles =quantiles, device = device, trainingScheme = trainingScheme , batchSize = batchSize)

    # remember that range(a,b) is actually [a,b) in python.
    predLabels = ['Task '+ str(i) for i in range(2, 16)] if tasks == "All" else ['Task '+ str(i) for
                                                                                 i in tasks]
    # Get the models, optimizer, train, evaluation and test (Task) data loader objects here.
    # Models are initilized with the same weights. The number of models returned is
    # a function of the combinization of optimizer parameters lr*moment*weight decay.
    # Note: Do not change.
    modelsList, optimTemplate, trainLoader, valLoader, testLoaders, schedule = init(**dataLoadArgs)

    # Model Train Invocation and other handling here
    args = []
    args.append(epochs)
    args.append(batchSize)
    args.append(schedule['labels'][0])  # Hold the Task label as a string i.e 'Task 4'. Used for annotation and saving.

    # List that holds all histories. 
    modelHistories = [modelsList[0].history for i in range(totalModels)]  
    evalPlotLabels = [args[2]]
    # For each parameter evaluate a model. Each models keep track of its history, and the optimizer
    # params with which it was trained.  
    idx = 0
    # TODO: Define validation, simple use and benchmark routines on top level.
    for l in range(len(gamma)):
        for m in range(len(momnt)):
            for w in range(len(wDecay)):
                for trainLoaders, tests, testLabels in zip(schedule['trainOn'], schedule['testOn'],
                                                          schedule['testLabels']):
                    # Use model in target position at a template
                    model = modelsList[idx].create_copy(device)
                    print("\nModel: {}@ {}. Lr: {} | Mom: {} | wDec: {}\n".format(idx,hex(id(model)), gamma[l], momnt[m],
                                                                      wDecay[w]))
                    # print(trainLoaders, tests)
                    print(len(trainLoaders), len(tests))
                    modelTrainInfo = 'Trained-on-'+ str(len(trainLoaders))

                    # Instantiate solver according to input params
                    optimParams = dict(name = optimName, params = dict(lr=[gamma[l]], momnt = [momnt[m]],
                                                                       wDecay=[wDecay[w]]))
                    optim = init_optim(model.parameters(), optimParams)

                    # Invoke training an Evaluation
                    model.train(args,device, trainLoaders, tests, optim, loss, saveHistory = True,
                                savePlot = True, modelLabel = modelTrainInfo, shuffleTrainLoaders =
                               True, saveRootFolder =scheme)
                    model.save(titleExt= '-trainedFor-'+str(epochs))
                    # ---|

                    # Predictions 
                    # NOTE: This evaluates the pretrained model on the selected tasks. Not yet online.
                    for i, loader in enumerate(schedule['predOn']):
                        print(schedule['predLabels'][i])
                        # args[2] = modelTrainInfo +'-tasks-for-'+str(epochs)+'-epochs-pred-on-'+ schedule['predLabels'][i][0]
                        args[2] = '-'.join((modelTrainInfo,'tasks-for',str(epochs),'epochs-pred-on', schedule['predLabels'][i][0]))
                        print(args[2])
                        model.predict(args, device, loader,lossFunction = loss, tarFolder =
                                      'Predictions/'+ modelTrainInfo, saveRootFolder = scheme)
                    # ---|

                    # Report saving and printouts go here
                    # print("Training history:")
                    # print(model.history)
                    modelHistories[idx] = model.history
                    args[2] = 'Train up to ' + testLabels[-1]  
                    evalPlotLabels.append(args[2])
                idx += 1
    # ---|
    # modelLabel = "0.5-0.7-0.1"
    modelLabel = scheme    # Plot total evaluation plot
    # This should become a function
    # keep in mind that join ignore blank strings
    filePath = join(dir_path, 'Logs', arch, modelLabel, 'PreTrain')
    f = plotter.get_files_from_path(filePath, "*preTrain-for-"+str(epochs)+"-epchs-log1.txt")
    print(f)
    files = []
    for i in f['files']:
        files.append(join(filePath, i)) 
    print("****\nPlotting Evaluation Curves...\n****")
    title = arch +' Learning Curves Evaluation\n Solid: Train, Dashed: Test'
    plotter.plot_regressor(filesPath = files, title = title, labels=evalPlotLabels)
    # plt.savefig(dir_path + '/Plots/' + arch +'/' + '/eval-plot-'+str(randint(0,20))+'.png')
    plotSavePath = join(dir_path, 'Plots', arch , modelLabel, 'eval-plot-'+str(randint(0,20))+'.png')
    print('Saving {} evaluation plots at: {}'.format(scheme, plotSavePath))
    plt.savefig(plotSavePath)
    plt.close()
#  End of main
#  -------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
