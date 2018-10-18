import sys as sys
import torch
import torchvision.datasets as tdata
import torchvision.transforms as tTrans

# WARNING:  This is relevant to the directory that CALLS this toplevel
# Module, rather than the position of this module.
sys.path.insert(0, './Code/')

from  Solvers import sgd
from Datasets import GEF_Power
from Architecture import MLR
from Tools import trainer, tester
print("Hello from power_GEF_14!")

# -------------------------------------------------------------------------------
# Data Loading Specifics --------------------------------------------------------

# Variable Definitions
batch = 1000

# ---|
 # path has to be relative from the directory of the file OR terminal
 # that calls this specific top level, not the actual location of
 # this top level itself.
load1 = GEF_Power.GefPower("./Data/GEF/Load/Task 1/L1-train.csv", transform = "normalize") 

device  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#TODO:  Check the data loader and get to work with integrating
#       the modular architecture!
comArgs = {'shuffle': True,'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

testLoader = torch.utils.data.DataLoader(load1, batch_size = batch, **comArgs)


# End of data loading -------------------------------------------------------
#----------------------------------------------------------------------------

print(len(testLoader.dataset))
print(load1.__getitem__(1))


#------------------------------------------------------------------------------
# Architecture specifics ------------------------------------------------------

# Model Declaration 
# model = forward_simple.Net().to(device)
model = MLR.LinearRegression(load1.get_item_size()).to(device)
print(model.get_model_descr())
# ---|




# Optimizer Declaration and paramater definitions fo here.
gamma = 0.0001
momnt = 0.5
optim = sgd.SGD(model.parameters(), weight_decay = 100, lr=gamma, momentum=momnt)
# ---|


# End of Architecture ---------------------------------------------------------
# -----------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Training Procedure ----------------------------------------------------------


# Variable Definitions
epochs = 2
# ---|

for e in range(epochs):
    args = e
    trainer.train_regressor(model, args, device, testLoader, optim)
    # tester.test(model, device, testLoader)

# Report
print(model.get_model_descr())
print(model.history)
# ---|

# End of Training --- ---------------------------------------------------------
# -----------------------------------------------------------------------------

