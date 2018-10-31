import sys as sys
import torch
import torchvision.datasets as tdata
import torchvision.transforms as tTrans
import os
import torch.nn as nn
import numpy as np

# WARNING:  This is relevant to the directory that CALLS this toplevel
# Module, rather than the position of this module.
dir_path   = os.path.dirname(os.path.realpath(__file__))
tools_path = os.path.join(dir_path, "../../Code/")
sys.path.insert(0, tools_path)


from  Solvers import sgd
from Datasets import GEF_Power
from Architecture import MLR, ann_forward, ann_greek
from Tools import trainer
print("Hello from power_GEF_14!")

def init(model = None, quantiles = [0.9], device = "cpu", trainDataRange = [0, 76799], testDataRange = [76800, 0], batchSize = 1000):
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
    outputSize = len(quantiles)
    print("Creating Model: {} at device: {}" .format(model, device))
    if model == "ANNGreek":
        model = ann_greek.ANNGreek(59).to(device)
    elif model == "MLRBIU":
        model = ann_forward.ANNLFS().to(device)
    elif model == "MRLSimple": 
        model = MLR.LinearRegression(25).to(device)
    elif "GLMLF" in model:
        model = MLR.GLMLFB(model, outputSize).to(device)
    print(model.get_model_descr())
    # ---|

    # -------------------------------------------------------------------------------
    # Data Loading Specifics --------------------------------------------------------

    # path has to be relative from the directory of the file OR terminal
    # that calls this specific top level, not the actual location of
    # this top level itself.
    #
    # Call this function to reshape raw data to match architecture specific inputs.
    # This function will reshape and save the data as: DataSet_reshaped_as_model.csv
    # delimitered by spaces.
    # GEF_Power.reshape_and_save("./Data/GEF/Load/Task 1/L1-train.csv", as = "ANNGReek") 
    dataPath = dir_path + "/../../Data/GEF/Load/Task 1/L1-train.csv"
    trainSet = GEF_Power.GefPower(dataPath, toShape = model.descr, transform =
                                  "normalize",dataRange= trainDataRange) 
    testSet = GEF_Power.GefPower(dataPath, toShape = model.descr, transform =
                                  "normalize",dataRange= testDataRange) 

    # Tell the Loader to bring back shuffled data, use 1 or more worker threads and pin-memory
    comArgs = {'shuffle': True,'num_workers': 2, 'pin_memory': True} if torch.cuda.is_available() else {}

    trainLoader = torch.utils.data.DataLoader(trainSet, batch_size = batchSize, **comArgs)
    testLoader = torch.utils.data.DataLoader(testSet, batch_size = batchSize, **comArgs)
    # Sanity prints
    print(len(testLoader.dataset))
    print(trainSet.__getitem__(0))

    return model, trainLoader, testLoader 
# ------------------------------------------------------------------------------------------------------------------
# Main Function 
# parameter and model selection, here.
def main():
    '''Description: This function is invoced then this top level is called.
                    It take the parsed arguments as input and will train,
                    teset and save the performance report of the endeavor.
                    The resulting plots are placed in the Plots folder.
                    The history is placed in the Logs folder.
    '''
    # ==========================================================================
    # Start of Parameter definitions
    # All Parameter choices done within the section defined by the thick seperator
    # lines

    # Variable Definitions
    epochs = 30 
    batchSize = 1000
    # quantiles = [0.01*i for i in range(1,100)]
    quantiles = [0.9]
    device  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Pass this dictionary as arg input to the init function. The data ranges should be relevant
    # To the raw data files input. All offsetting etc is taken care of from the dataset code
    dataLoadArgs  = dict(model = "GLMLF-C2", quantiles = quantiles, device = device, trainDataRange
                         = [0, 76799], testDataRange = [76800, 0], batchSize = batchSize)
    # File, plot and log saving variables. Leaveto None, to save to default locations
    logSavePath  = None
    plogSavePath = None
    # ---|

    # Get the model, and train and test data loader objects here
    # Note: Do not change.
    model, trainLoader, testLoader = init(**dataLoadArgs)

    # Optimizer Declaration and parameter definitions go here.
    gamma = 0.1
    momnt = 0.5
    optim = sgd.SGD(model.parameters(), weight_decay = 0.1, lr=gamma, momentum=momnt)
    # ---|

    # Loss Function Declaration and parameter definitions go here.
    loss = trainer.QuantileLoss(quantiles)
    # loss = nn.MSELoss()
    # ---|

    # End of parameter Definitions. Do not alter below this point.
    # ==========================================================================

    # Model Train Invocation and other handling here
    args = []
    args.append(epochs)
    args.append(batchSize)

    # Invoke training an Evaluation
    model.train(args,device, trainLoader, testLoader,optim, loss)
    # ---|

    # Report saving and printouts go here
    print("Learned model:\n" + model.get_model_descr())
    print("Training history:")
    print(model.history)
    model.save_history(logSavePath)
    model.plot()
    titleExt = optim.name + "-lr-" +  str(optim.lr) + "-momnt-" + str(optim.momnt)
    model.save_plots(plogSavePath, titleExt)
    # ---|

#  End of main
#  -------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
