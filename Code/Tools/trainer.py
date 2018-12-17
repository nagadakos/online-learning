
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optm
import sys
import os 
import numpy as np
dir_path = os.path.dirname(os.path.realpath(__file__))

sys.path.insert(0, dir_path)

import regression_idx as ridx
from Tools.utils import QuantileLoss


def save_log(filePath, history):
    '''
        Description: Saves the history log in the target txt file.
                     If some history elements do not exist, mark them with -1.
        Arguments:   filePath (string): Target location for log
                     history (list of lists): History list in the following format:
                     Each  inner list is one of trainMAE, testLOss etc, as indexed 
                     in the ridx file. They contain the relevant metric from all epochs
                     of training / testing, if they exist.
    '''
    
    with open(filePath, 'w') as f:
        for i in range(len(history[0])): 
            for j in range(len(history)):
                if j < len(history):
                    f.write("{:.4f} ".format(history[j][i]))
                else:
                    f.write("-1")
            f.write("\n")
     
# End of Save Log.
# ----------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------
# Training function for MLR Regressors

def train_regressor(model, args, device, indata, optim, lossFunction = nn.MSELoss()):
    '''
        Description: This function tests regression models. It computes
                     the Mean Absolute Error (MAE) and Mean Absolute Relative Error(MAPE).
                     If other metric are needed fille the logic in the marked place.
                     Also, one needs to edit the regression_idx file, and update the
                     indexes of the new metrics. Every architecture's history is indexed by
                     those indexes.

        Arguments:  model (nn.module):        A specified architecture, to be trained.
                    args [list]:              List of various arguments. 
                                              Curently 0: epoch  1: batchsize
                    Device (device):          Selects CPU or GPU utilization
                    testloader (dataLoader):  Data loader objest for data... loading
                    lossFunction (nn.module): Desired loss function. Default is MSE.help
    '''
    MAE   = 0
    MAPE  = 0
    e = 0.001     # used for avoiding division by 0.
    totalSize = 0 # used for averaging error in the end
    factor = 0    # used for AVG MAE display  
    if not isinstance(indata, list):
        indata = [indata]

    for setID, dSet in enumerate(indata):
        print(dSet)
        for idx, (data, label) in enumerate(dSet):
            data, label = data.to(device), label.to(device)
            # forward pass calculate output of model
            output = model.forward(data)
            # Reshape data and labels from (n*modelOutSize,) and (n,) to (n, modelOutSize) and (n,1)
            pred   = output.view(len(label), output.shape[1])
            label  = label.view(len(label), 1)
            # Sanity prints
            # if idx == 5:
                # # print("Prediction shape: {} label shape: {}". format(pred.shape, label.shape))
                # print("Prediction: {}, labels: {}" .format(pred, label))

            # compute loss
            if isinstance(lossFunction, QuantileLoss):
                loss, lossMatrix = lossFunction.forward(pred, label)
            else:
                loss = lossFunction(pred, label)

            MAE  += torch.FloatTensor.abs(pred.sub(label)).sum().item()
            MAPE += torch.FloatTensor.abs(pred.sub(label)).div(label+e).sum().item()

            # Backpropagation part
            # 1. Zero out Grads
            optim.zero_grad()
            # 2. Perform the backpropagation based on loss
            loss.backward()            
            # 3. Update weights 
            optim.step()
           # Training Progress report for sanity purposes! 
            if idx % 4 == 0 or idx % pred.shape[0] == 0 : 
                print("Epoch: {}-> Batch: {} / {}, Size: {}. Loss = {}".format(args, idx, len(dSet),
                                                                               pred.shape[0], loss.item() ))
                factor += pred.shape[0]
                print("Average MAE: {}, Average MAPE: {:.4f}%".format(MAE / factor, MAPE*100 /factor))
        totalSize += len(dSet.dataset)
    # Log the current train loss
    MAE  = MAE/totalSize
    MAPE = MAPE*100/totalSize
    model.history[ridx.trainLoss].append(loss.item())   #get only the loss value
    model.history[ridx.trainMAE].append(MAE)
    model.history[ridx.trainMAPE].append(MAPE)
#--------------------------------------------------------------------------------------
# Start of Train Regressor.

def test_regressor(model, args, device, testLoader, trainMode= False, lossFunction = nn.MSELoss()):

    MAE  = 0 
    MAPE = 0
    loss = 0
    totalSize = 0
    batchSize = args[1]
    # Inform Pytorch that keeping track of gradients is not required in
    # testing phase.
    with torch.no_grad():

        if not isinstance(testLoader, list):
            testLoader = [testLoader]
        for setID, dSet in enumerate(testLoader):
            for data, label in dSet:
                data, label = data.to(device), label.to(device)
                output = model.forward(data)
                # Reshape data and labels from (n*modelOutSize,) and (n,) to (n, modelOutSize) and (n,1)
                pred   = output.view(len(label), output.shape[1])
                label  = label.view(len(label), 1)

                if isinstance(lossFunction, QuantileLoss):
                    # Sum all loss terms and tern then into a numpy number for late use.
                    loss, lossMatrix = lossFunction(pred, label)
                    loss= loss.item()
                else:
                    loss = lossFunction(pred, label).item()
                # TODO: Properly define MAE and MAPE in Quantile Regression setting.
                MAE  += torch.FloatTensor.abs(pred.sub(label)).sum().item()
                MAPE += torch.FloatTensor.abs(pred.sub(label)).div(label).mul(100).sum().item()
            totalSize += len(dSet.dataset)
        
        # Log the current train loss
    MAE  = MAE/ totalSize
    MAPE = MAPE/totalSize  
    loss = loss 
    

    # Print Regressor's evaluation report!
    if trainMode == True:
        print("--Epoch {}  Testing --".format(args[0]))
        print("Average MAE: {}, Average MAPE: {:.4f}%, Agv Loss: {:.4f}".format(MAE, MAPE, loss))
        print("-------")
        model.history[ridx.testLoss].append(loss)   #get only the loss value
        model.history[ridx.testMAE].append(MAE)
        model.history[ridx.testMAPE].append(MAPE)
        
    if trainMode == False:
        print("Predicting on {}".format(args[2]))
        print("Average MAE: {}, Average MAPE: {:.4f}%, Agv Loss: {:.4f}".format(MAE, MAPE, loss))
        print("-------")
        model.predHistory[0].append(0)      # fill with 0, for plotting compatibility
        model.predHistory[1].append(0)
        model.predHistory[2].append(0)
        model.predHistory[ridx.predLoss].append(loss)   #get only the loss value
        model.predHistory[ridx.predMAE].append(MAE)
        model.predHistory[ridx.predMAPE].append(MAPE)

    return pred, loss, lossMatrix

# End of train Regressor 
# ---------------------------------------------------------------------------------------_

# ------------------------------------------------------------------------------------------------------
# Main funcion: Used for debuging logic.
# ------------------------------------------------------------------------------------------------------
def main():
    history = [[1, 2],[3,4],[5,6],[7,8],[9,10],[]]
    print(len(history))
    filePath = "./log1.txt"
    save_log(filePath, history)


if __name__ == "__main__":
    main()
