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


from  Solvers import sgd, time_smoothed_sgd,time_smoothed_sgd2
from Datasets import GEF_Power
from Architecture import MLR, ann_forward, ann_greek
from Tools import trainer,plotter,utils

# ================================================================================================================
# Start of Functions
# ================================================================================================================

def init_optim(modelParams, optimParams = dict(name="SGD", params=dict(lr=0.1,momnt=0.5,wDecay=0.9, w =1, a =0.5, lrScheme = 'constant'))):
    ''' Description: This function initializes an optimizer obect according to input parameters.
        
        Arguments: 1.modelParams(PyTorch params) The model parameters iterator to be optimized.
                   2. optimParams(dict):    Dictionary with args. Name = Name of solver. Params = dict of optimizer mathematical 
                                            parameters such as learning rate.
        Returns:    optimTemplate: An solver object with the required parameters and ready to optimize the given moel's params.

    '''
    print(optimParams['params']['lrScheme'][0])
    # Initialize the optimizer Template with the parameters of
    if optimParams["name"] == "SGD":
        optimTemplate = sgd.SGD(modelParams, weight_decay = optimParams["params"]["wDecay"][0],
                                lr=optimParams["params"]["lr"][0], momentum=optimParams["params"]["momnt"][0])
    elif "TSSGD" in optimParams["name"]:
        optimTemplate = time_smoothed_sgd.TSSGD(modelParams, name = optimParams['name'],weight_decay =   
                                                 optimParams["params"]["wDecay"][0], lr=optimParams["params"]["lr"][0], 
                                                 momentum=optimParams["params"]["momnt"][0],
                                                 w=optimParams["params"]["w"][0], a=optimParams["params"]["a"][0],
                                                 lrScheme = optimParams['params']['lrScheme'][0])
    else:
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        optimTemplate = sgd.SGD(modelParams, weight_decay = optimParams["params"]["wDecay"][0],
                                lr=optimParams["params"]["lr"][0], momentum=optimParams["params"]["momnt"][0])

    return optimTemplate

# End of init_optim
#-----------------------------------------------------------------------------------------------------
def load_data(preTrainYears, dataPath = None, batchSize = 1000,tasks = 'All', loadFromEnd = False, reshapeDataTo = None,
              val_percent = 0.2, startFrom =0):
    ''' Description: This function will handle the creation of dataloaders for the scheduler to arrange into valid training
                     schedules. Trainloader and ValLoader are in all cases used for the preTrain part of the process. Task
                     loaders are  the dataset files to be used for prediction and online learning.

        Arguments:  1. preTrainYears(int): The number of years of data to be used for preTrain.
                    2. dataPath(filePath): The file path of the data set, if it resides outside the default location.
                    3. batchSize(int):     Batchsize for data loading.
                    4. tasks(list(int)):   A list of ints indigating which specific tasks to load. Default is All.
                    5. loadFromEnd(Bool):  Flag for load start point. False means start loading from
                                           start of original dataset. True meansstart loading data from the end towards the
                                           start.
                    6. reshapeDataTo(str): A string used to tell the dataloader class to reshape the data according to the
                    7. vla_percent(float): A number 0-1 that selects the percentage of the set used as validation.
                                           requirements of each architecture. Should be defined in the Dataset class first.
                    8. startFrom(int):     A number 0-1 that ingore the given percentage of the dataset from start. 
        Returns:    trainLoader(dataLoader) A dataloader object that contains the preTrain data.
                    ValLoader:              A dataload object that contains preTrain validation data.
                    TaskLoaders:            Dataloader Object that contains the Task data.hasattr
                    filesNum:               A list of ints, containing the loaded task files numbers (i.e 2 for Task 2 etc)
    '''
    # Tell the Loader to bring back shuffled data, use 1 or more worker threads and pin-memory
    comArgs = {'shuffle': True,'num_workers': 2, 'pin_memory': True} if torch.cuda.is_available() else {}

    # Load all the 15 files as taskloaders. Each one is one of the 15 tasks to assess learning on.
    # If required return the task loaders for the online schemes.
    filesNum = tasks if tasks != "All" else [1*i for i in range(2,16)]
    taskLoaders = [[],[]]
    for i, t in enumerate(filesNum):
        start = startFrom 
        testSet = GEF_Power.GefPower(dataPath, task ='Task ' +str(t), dataRange=[start,1-val_percent], toShape = reshapeDataTo, transform = "normalize") 
        testLoader = torch.utils.data.DataLoader(testSet, batch_size = batchSize, **comArgs)
        taskLoaders[0].append(testLoader)
        valSet = GEF_Power.GefPower(dataPath, task ='Task ' +str(t), dataRange=[1-val_percent,0], toShape = reshapeDataTo, transform = "normalize") 
        valLoader = torch.utils.data.DataLoader(testSet, batch_size = batchSize, **comArgs)
        taskLoaders[1].append(valLoader)
        # print(len(testSet))

    # Task 1 file offset.
    if loadFromEnd == True:
        trainStart = 85443 - int(preTrainYears * 365.25)*24
        trainEnd = 85443 - int(preTrainYears * 365.25*24 * val_percent)
        trainDataRange = [trainStart, trainEnd]
        testDataRange = [trainEnd+1,0] # 0 means till EOF
    else:
        trainDataRange = [0, 35064 + int(preTrainYears * 365.25)*24] #if trainingScheme['update'] != 'Benchmark' else [0, 90000]
        testDataRange = [trainDataRange[1], 0]
    print(trainDataRange)

    trainSet = GEF_Power.GefPower(dataPath, toShape = reshapeDataTo, transform =
                                  "normalize",dataRange= trainDataRange) 
    valSet = GEF_Power.GefPower(dataPath, toShape = reshapeDataTo, transform =
                                  "normalize",dataRange= testDataRange) 

    
    trainLoader = torch.utils.data.DataLoader(trainSet, batch_size = batchSize, **comArgs)
    valLoader = torch.utils.data.DataLoader(valSet, batch_size = batchSize, **comArgs)

    return trainLoader, valLoader, taskLoaders, filesNum

# End of load_data
#-----------------------------------------------------------------------------------------------------

def init(model = None, tasks = "All", quantiles = [0.9], device = "cpu", batchSize = 1000, val_percent = 0.2, startFrom=0,
         optimParams = dict(name="SGD", params=dict(lr=0.1,momnt=0.5,wDecay=0.9,w=1,a=0.5, lrScheme = 'constant')), 
         trainingScheme = dict(preTrainOn = ['5 Years'] , update ='Benchmark', loadFromEnd= False, lrScheme = 'constant', 
                               trainStopage= 'fixed', preTrainOptim = ['SGD'], updateOptim = ['SGD'], epochScheme = [25,25])):
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
    windows = optimParams['params']['w']
    alphas  = optimParams['params']['a']
    # Compute the total num of models. Might be usefull.
    numOfModels = len(lrs)*len(momnts)*len(wDecays)
    outputSize = len(quantiles)
    idx = 0
    for l in range(len(lrs)):
        for m in range(len(momnts)):
            for w in range(len(wDecays)):
                for wi in range(len(windows)):
                    for a in range(len(alphas)):
                        if model == "ANNGreek":
                            reshapeDataTo = model
                            models.append( ann_greek.ANNGreek(59, outputSize, optimParams['name'],
                                                              lr=lrs[l],momnt=momnts[m], wDecay=wDecays[w]
                                                             ,w =windows[wi]).to(device))
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
    loadFromEnd = trainingScheme['loadFromEnd']
    preTrainYears = float(trainingScheme['preTrainOn'][0].split()[0])
    print("PRETRAIN years" + str(preTrainYears))
    # Call this function to reshape raw data to match architecture specific inputs.
    # This function will reshape and save the data as: DataSet_reshaped_as_model.csv
    # delimitered by spaces.
    #
    # Will create data loaders for: pretrain, validation, prediction tasks and a list of pred tasks
    # numbers
    trainLoader, valLoader, taskLoaders, filesNum = load_data(preTrainYears, loadFromEnd = loadFromEnd, batchSize= batchSize, 
                                                              reshapeDataTo = reshapeDataTo, val_percent = val_percent, 
                                                              startFrom = startFrom)

    # Sanity prints
    print(len(taskLoaders[0][0].dataset))
    # print(testSet.__getitem__(0))
    
    # ------------------------------------------------------------------------------    -
    # SCHEDULER
    # *********

    # Generate a schedule of training and testing schemes.
    schedule = dict( trainOn = [], testOn = [], predOn = [], labels =[], testLabels =[], predLabels
                   = [], optim= [], lrUpdate =[], trainStop = [], epchScheme = [])
    # Benchmark creates a benchmark case for online training.
    # It uses naively, all available data up to a point to train, and 
    # tests on the rest. I.e train on Task1, 2 ,3 and test on task 4: end.
    # The trainset start from Task 1, Task 1 + val, Task 1 + val + Task 2...
    # The test set Val + Task loaders, Taskloaders, TaskLoaders[2:end] ...
    # The predSet has to be a list of lists. It can have just one list.
    preTrainOptim = trainingScheme['preTrainOptim']
    updateOptim = trainingScheme['updateOptim']
    schedule['optim'].append(preTrainOptim)
    schedule['epchScheme'].append(trainingScheme['epochScheme'][0])
    schedule['trainStop'].append(trainingScheme['trainStopage'][0])
    if trainingScheme['update'] == 'Benchmark':
        schedule['trainOn'].append([trainLoader])
        schedule['testOn'].append([valLoader])
        schedule['labels'].append(['Task 1'])
        schedule['predOn'].append([taskLoaders[0][0]])
        schedule['predLabels'].append(['Pred on Task ' + str(filesNum[0])])
        schedule['testLabels'].append(['Task ' + str(filesNum[0])])
        for i, t in enumerate(filesNum[:-1]):
            print(i, t)
            prevTrain = schedule['trainOn'][i].copy()
            prevLabels  = schedule['labels'][i].copy()
            prevTrain.append(taskLoaders[0][i])
            prevLabels.append('Task '+str(i+2))
            schedule['trainOn'].append(prevTrain)
            schedule['testOn'].append([taskLoaders[1][i]])
            schedule['labels'].append(prevLabels)
            schedule['testLabels'].append(['Task ' + str(filesNum[i+1])])
            schedule['predOn'].append([taskLoaders[0][i+1]])
            schedule['predLabels'].append(['Pred on Task ' + str(filesNum[i+1])])
            schedule['optim'].append(preTrainOptim)
            schedule['epchScheme'].append(trainingScheme['epochScheme'][1])
            schedule['trainStop'].append(trainingScheme['trainStopage'][1])
    elif trainingScheme['update'] == 'Online':
        schedule['trainOn'].append([trainLoader])
        schedule['testOn'].append([valLoader])
        schedule['labels'].append(['Task 1'])
        schedule['testLabels'].append(['Task ' + str(filesNum[0])])
        schedule['predOn'].append([taskLoaders[0][0]])
        schedule['predLabels'].append(['Pred on Task ' + str(filesNum[0])])
        for i, t in enumerate(filesNum[:-1]):
            print(i, t)
            schedule['trainOn'].append([taskLoaders[0][i]])
            schedule['testOn'].append([taskLoaders[1][i]])
            schedule['labels'].append(['Trained-up-to-task-' + str(filesNum[i])])
            schedule['testLabels'].append(['Task-' + str(filesNum[i])])
            schedule['predOn'].append([taskLoaders[0][i+1]])
            schedule['predLabels'].append(['Pred on Task ' + str(filesNum[i+1])])
            schedule['optim'].append(updateOptim)
            schedule['epchScheme'].append(trainingScheme['epochScheme'][1])
            schedule['trainStop'].append(trainingScheme['trainStopage'][1])
    elif trainingScheme['update'] == 'ParamEvaluation':
        schedule['trainOn'].append([trainLoader])
        schedule['testOn'].append([valLoader])
        trainOrder = 'first' if loadFromEnd == False else 'last'
        schedule['labels'].append(['-'.join(('param-eval-trainedOn',trainOrder, str(preTrainYears)))])
        schedule['testLabels'].append(['EvalSet ' + str(5-preTrainYears)])
        schedule['predOn'].append([])
    elif trainingScheme['update'] == 'Offline':
        schedule['trainOn'].append([trainLoader])
        schedule['testOn'].append([valLoader])
        trainOrder = 'first' if loadFromEnd == False else 'last'
        schedule['labels'].append(['-'.join(('Task 1-OffLine-trainedOn',trainOrder, str(preTrainYears)))])
        schedule['testLabels'].append(['EvalSet ' + str(5-preTrainYears)])
        # Only one pred task. just provide a results for each one of the tasks.
        schedule['predOn'].append(taskLoaders[0])
        schedule['optim'].append(preTrainOptim)
        for i, t in enumerate(filesNum):
            print(i, t)
            schedule['predLabels'].append(['Pred on Task ' + str(filesNum[i])])
    elif trainingScheme['update'] == 'Default':
        schedule['trainOn'].append([trainLoader])
        schedule['testOn'].append([valLoader])
        schedule['labels'].append(['Task 1-Default'])
        schedule['testLabels'].append(['EvalSet ' + str(5-preTrainYears)])
        schedule['predOn'].append([taskLoaders[0]])
        labels = []
        for i, t in enumerate(filesNum):
            print(i, t)
            labels.append(['Pred on Task ' + str(filesNum[i])])
        schedule['predLabels'].append(labels)
    # print(trainingScheme['epochScheme'])
    # print(schedule)
    # print(schedule['testOn'])
    # print(schedule['testLabels'])
    # print(schedule['predOn'])
    # print(schedule['predLabels'])
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
    preTrainEpochs = 50         # must be at least 2 for plot with labellines to work
    updateEpochs = 30
    batchSize = 1000
    val_percent = 0.2             # percentage of each available set to use as validation.
    startFrom = 0.0
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
    gamma = [0.9] # learning rate
    momnt = [0.5]#, 0.5] # momentum
    wDecay= [0.01]       # weight decay (l2 normalization)   
    window= [1]          # window size for time smoothed variants
    # alpha = [0,0.1,0.3,0.5,0.7,0.9]
    alpha = [1]
    preTrainOptimName = "SGD"    # SGD,(DW)-A-TSSGD, (DW-)ED-TSSGD
    updateOptimName =  'SGD' # only affects the online case
    lrScheme = ['constant']
    trainStopage = ['fixed', 'adaptive'] # Stopagge cheme for 0: preTrain, 1: update. Options: adaptive, fixed
    totalModels = len(gamma) * len(momnt) * len(wDecay) * len(window) * len(alpha)
    # ---|

    # Task loading and online learning params go here.
    tasks = 'All' # Use this to load all Tasks 
    #                    0               1              2          3          4
    availSchemes = ['Benchmark', 'ParamEvaluation', 'Default', 'Offline', 'Online']
    scheme = availSchemes[0] # Benchmark, ParamEvaluation
    epochScheme = [preTrainEpochs, updateEpochs]
    # Do this to see if withholding data will affect the accuracy from the onlinecase.
    if scheme == 'Online':
        startFrom =0.5
    # set has 5.75 years. Update: monthly, weekly. Load from end tells loader to load data from the
    # last years to the first.
    trainingScheme= dict(preTrainOn = ["4 Years"], update = scheme, loadFromEnd = True, lrScheme = lrScheme, trainStopage =
                         trainStopage, preTrainOptim = preTrainOptimName, updateOptim = updateOptimName, epochScheme = epochScheme)   
                                                                         # benchmark
    # tasks = [2, 4, 6, 8] # Use this to provide a list of the required subset!
    # ---|

    # File, plot and log saving variables. Leaveto None, to save to default locations
    plotResults = 0
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
    optimParams = dict(name = preTrainOptimName, params = dict(lr=gamma, momnt = momnt, wDecay=wDecay, w = window, a = alpha, lrScheme = lrScheme))
    dataLoadArgs  = dict(model = arch, tasks = tasks, optimParams = optimParams, quantiles =quantiles, device = device, 
                         trainingScheme = trainingScheme , batchSize = batchSize, val_percent= val_percent, startFrom=startFrom)
    
    
    # remember that range(a,b) is actually [a,b) in python.
    # predLabels = ['Task '+ str(i) for i in range(2, 16)] if tasks == "All" else ['Task '+ str(i) for i in tasks]

    # Add extra label for Time-smoothed SGD trainer
    optimUpdateLabel = '-'.join((lrScheme[0]+'LR' , trainStopage[0],trainStopage[1], 'stop'))
    # Get the pretrain specifics
    preTrainNum = int(trainingScheme['preTrainOn'][0][0])
    preTrainType = trainingScheme['preTrainOn'][0][2:]
    # Get the models, optimizer, train, evaluation and test (Task) data loader objects here.
    # Models are initilized with the same weights. The number of models returned is
    # a function of the combinization of optimizer parameters lr*moment*weight decay.
    # Note: Do not change.
    modelsList, optimTemplate, trainLoader, valLoader, testLoaders, schedule = init(**dataLoadArgs)

    # Model Train Invocation and other handling here
    args = []
    args.append(schedule['epchScheme'][0])
    args.append(batchSize)
    args.append(schedule['labels'][0])  # Hold the Task label as a string i.e 'Task 4'. Used for annotation and saving.

    # List that holds all histories. 
    modelHistories = [modelsList[0].history for i in range(totalModels)]  
    evalPlotLabels = [args[2]]
    # For each parameter evaluate a model. Each models keep track of its history, and the optimizer
    # params with which it was trained.  
    idx = 0
    for l in range(len(gamma)):
        for m in range(len(momnt)):
            for wD in range(len(wDecay)):
                for wi in range(len(window)):
                    for a in range(len(alpha)):
                        # Use model in target position at a template
                        model = modelsList[idx].create_copy(device)
                        print("\nModel: {}@ {}. Lr: {} | Mom: {} | wDec: {}\n".format(idx,hex(id(model)), gamma[l], momnt[m],
                                                                              wDecay[wD]))
                        # Instantiate solver according to input params
                        optimParams = dict(name = preTrainOptimName, params = dict(lr=[gamma[l]], momnt = [momnt[m]],
                                      wDecay=[wDecay[wD]], w =[window[wi]], a=[alpha[a]], lrScheme=lrScheme))
                        optim = init_optim(model.parameters(), optimParams)
                        # Train according to schedules here. trainloaders and tests can be lists of dataloaders.
                        for sIdx, (trainLoaders, tests, testLabels) in enumerate(zip(schedule['trainOn'], schedule['testOn'],
                                                                  schedule['testLabels'])):
                            # --------------------------------------------------------------------
                            # EXTRACT SCHEDULER INFO
                            # ******
                            # use the optimizer specified by the schedule.
                            # if it is different than the current one, init the new optimizer.
                            if optimParams['name'] != schedule['optim'][sIdx]: 
                                print("CHanging optim from {} to {}". format(optimParams['name'], schedule['optim'][sIdx]))
                                optimParams['name'] = schedule['optim'][sIdx]
                                # NOTE: This is to ensure no momentum is used for the time smoothed tests, untill the parameter are also given by
                                # the  schedule.
                                optimParams['params']['momnt']= [0.5] if (optimParams['name'] =='SGD' and sIdx == 0 
                                                                          and momnt[m] != 0) else [0]
                                optim = init_optim(model.parameters(), optimParams)
                            # ---|
                            # Training stoppage scheme to feed into the trainer function.
                            adaptStopSel = True if schedule['trainStop'][sIdx] == 'adaptive' else False
                        
                            # --------------------------------------------------------------------

                            # Form model label string for save purposes
                            modelTrainInfo = '-'.join((schedule['optim'][1],optimUpdateLabel,str(alpha[a]),str(window[wi])))
                            schemeSpecific = ''
                            if trainingScheme['update'] == 'Benchmark':
                                schemeSpecific = '-'.join(('Trained-on', str(len(trainLoaders)), 'Tasks'))
                            modelTrainInfo = '-'.join((modelTrainInfo, schemeSpecific,'preTrain-on',str(preTrainNum)
                                                       ,preTrainType,'with',schedule['optim'][0]))
                            # ---|

                            # Invoke training and Evaluation
                            args[0] = schedule['epchScheme'][sIdx]  # This holds the max epochs to train for the train function. TODO: turn
                                                                    # iti inti discionary
                            model.train(args,device, trainLoaders, tests, optim, loss, saveHistory = True, savePlot = False, 
                                        modelLabel = modelTrainInfo, shuffleTrainLoaders = False, saveRootFolder =scheme,
                                        adaptDurationTrain = adaptStopSel)
                            trainDurInfo = '-'.join(('-trainedFor',str(args[0]),'epchs')) # form a train dur string, for logging
                            model.save(titleExt = trainDurInfo)
                            # ---|

                            # Predictions 
                            # The scheduled predicitons might be on a lot of datasets. The results will be logged
                            # in the same file, one line after another.
                            for i, loader in enumerate(schedule['predOn'][sIdx]):
                                print(schedule['predLabels'][i])
                                args[2] = '-'.join(('-pred-on', schedule['predLabels'][sIdx][0]))
                                print(args[2])
                                model.predict(args, device, loader,modelLabel= modelTrainInfo+trainDurInfo, lossFunction = loss,  
                                              tarFolder ='Predictions/'+ modelTrainInfo, saveRootFolder = scheme, saveResults = 
                                              False)
                            # ---|

                            # Report saving and printouts go here
                            # print("Training history:")
                            # print(model.history)
                            modelHistories[idx] = model.history
                            args[2] = 'Train up to ' + testLabels[-1]  
                            evalPlotLabels.append(args[2])
                            # Benchmark case needs to be retrained from scratch every time.
                            if trainingScheme['update'] == 'Benchmark':
                                tempHistory = model.predHistory
                                model = modelsList[idx].create_copy(device)
                                optim = init_optim(model.parameters(), optimParams)
                                model.predHistory = tempHistory
                        idx += 1
    # ---|
    if plotResults:
        modelLabel = scheme    # Plot total evaluation plot
        # This should become a function
        # keep in mind that join ignore blank strings
        filePath = join(dir_path, 'Logs', arch, modelLabel, 'PreTrain')
        f = plotter.get_files_from_path(filePath, "*for-"+str(epochs)+"-epchs-log1.txt")
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
