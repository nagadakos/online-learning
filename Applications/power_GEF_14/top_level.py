import sys as sys
import torch
import torchvision.datasets as tdata
import torchvision.transforms as tTrans
import os
import torch.nn as nn


# WARNING:  This is relevant to the directory that CALLS this toplevel
# Module, rather than the position of this module.
sys.path.insert(0, './Code/')
dir_path = os.path.dirname(os.path.realpath(__file__))



from  Solvers import sgd
from Datasets import GEF_Power
from Architecture import MLR, ann_forward
from Tools import trainer
print("Hello from power_GEF_14!")

# -------------------------------------------------------------------------------
# Data Loading Specifics --------------------------------------------------------

# Variable Definitions
batch = 1000

# ---|
 # path has to be relative from the directory of the file OR terminal
 # that calls this specific top level, not the actual location of
 # this top level itself.

# Call this function to reshape raw data to match architecture specific inputs.
# This function will reshape and save the data as: DataSet_reshaped_as_model.csv
# delimitered by spaces.
# GEF_Power.reshape_and_save("./Data/GEF/Load/Task 1/L1-train.csv", as = "ANNGReek") 
trainSet = GEF_Power.GefPower("./Data/GEF/Load/Task 1/L1-train.csv", transform =
                              "normalize",dataRange= [0,76799]) 
testSet = GEF_Power.GefPower("./Data/GEF/Load/Task 1/L1-train.csv", transform =
                             "normalize",dataRange=[76800,0]) 

device  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#TODO:  Check the data loader and get to work with integrating
#       the modular architecture!
comArgs = {'shuffle': True,'num_workers': 2, 'pin_memory': True} if torch.cuda.is_available() else {}

trainLoader = torch.utils.data.DataLoader(trainSet, batch_size = batch, **comArgs)
testLoader = torch.utils.data.DataLoader(testSet, batch_size = batch, **comArgs)


# End of data loading -------------------------------------------------------
#----------------------------------------------------------------------------

print(len(testLoader.dataset))
# print()
print(trainSet.__getitem__(0))


#------------------------------------------------------------------------------
# Architecture specifics ------------------------------------------------------

# Model Declaration 
# model = forward_simple.Net().to(device)
model = MLR.LinearRegression(25).to(device)
# model = ann_forward.ANNLFS().to(device)
print(model.get_model_descr())
# ---|




# Optimizer Declaration and paramater definitions fo here.
gamma = 0.001
momnt = 0.5
optim = sgd.SGD(model.parameters(), weight_decay = 3, lr=gamma, momentum=momnt)
# ---|


# End of Architecture ---------------------------------------------------------
# -----------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Training Procedure ----------------------------------------------------------


# Variable Definitions
epochs = 12
# ---|
args = []
args.append(epochs)
args.append(batch)
loss = trainer.QuantileLoss(0.9)
# loss = nn.MSELoss()
model.train(args,device, trainLoader, testLoader,optim, loss)


# Report
print(model.get_model_descr())
print(model.history)
logFilePath = dir_path + "/Logs/log1.txt"
model.save_history(logFilePath)
# ---|

# End of Training --- ---------------------------------------------------------
# -----------------------------------------------------------------------------

