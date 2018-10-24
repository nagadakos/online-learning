
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optm
import sys
import os 

dir_path = os.path.dirname(os.path.realpath(__file__))

sys.path.insert(0, dir_path)

import classification_idx as indexes
import regression_idx as ridx



class QuantileLoss(nn.Module):
    '''
        Description: This module models Quantile loss. it is implmented as a nn module
                     So as to enable backproagation to compute gradients efficiently.

        Returns:    x (loss): Averaged quantile loss over all input elements.
    '''
    def __init__(self,q):
        super(QuantileLoss, self).__init__()
        self.q = q

    def forward(self, x, target):
        x = self.q* F.relu(x-target) + (1-self.q) * F.relu(target- x)
        x = x.mean() # Mean value of all elements, scalar.
        return x
# End of Quantile Loss
# ---------------------------------------------------------------------------------


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
                if i < len(history[j]):
                    f.write("{:.4f} ".format(history[j][i]))
                else:
                    f.write("-1")
            f.write("\n")
     
# End of Save Log.
# ----------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------
# Start of Train
# This function is aimed towards training classifiers. For training regressors
# see train_regressor instead.

def train(model, args, device, indata, optim, lossFunction = nn.MSELoss()):
    true = 0
    acc  = 0
    for idx, (img, label) in enumerate(indata):
        data, label = img.to(device), label.to(device)
        # forward pass calculate output of model
        output      = model.forward(data)
        pred   = output.max(dim = 1, keepdim = True)
        true  += label.eq(pred[1].view_as(label)).sum().item()
        # print(data.shape)
        # print(output.shape)
        # compute loss
        loss        = lossFunction(output, label)
        # Backpropagation part
        # 1. Zero out Grads
        optim.zero_grad()
        # 2. Perform the backpropagation based on loss
        loss.backward()            
        # 3. Update weights 
        optim.step()

       # Training Progress report for sanity purposes! 
        # if idx % 20 == 0: 
            # print("Epoch: {}->Batch: {} / {}. Loss = {}".format(args, idx, len(indata), loss.item() ))
    # Log the current train loss
    acc = true/len(indata.dataset)
    model.history[indexes.trainLoss].append(loss.item())   #get only the loss value
    model.history[indexes.trainAcc].append(acc)

# ------------------------------------------------------------------------------------------
# Training function for MLR Regressors

def train_regressor(model, args, device, indata, optim, lossFunction = nn.MSELoss()):
    MAE   = 0
    MAPE  = 0

    for idx, (data, label) in enumerate(indata):
        data = model.shape_input(data)     # select specified columns only.
        data, label = data.to(device), label.to(device)
        # forward pass calculate output of model
        output = model.forward(data).view_as(label)
        pred   = output
        
        # Sanity prints
        # if idx == 5:
            # # print("Prediction shape: {} label shape: {}". format(pred.shape, label.shape))
            # print("Prediction: {}, labels: {}" .format(pred, label))

        # compute loss
        loss = lossFunction.forward(pred, label)
        MAE  += torch.FloatTensor.abs(pred.sub(label)).sum().item()
        MAPE += torch.FloatTensor.abs(pred.sub(label)).div(label).mul(100).sum().item()
        # Backpropagation part
        # 1. Zero out Grads
        optim.zero_grad()
        # 2. Perform the backpropagation based on loss
        loss.backward()            
        # 3. Update weights 
        optim.step()
       # Training Progress report for sanity purposes! 
        if idx % 20 == 0 or idx % pred.shape[0] == 0 : 
            print("Epoch: {}-> Batch: {} / {}, Size: {}. Loss = {}".format(args, idx, len(indata),
                                                                           pred.shape[0], loss.item() ))
            factor = (idx+1)*pred.shape[0]
            print("Average MAE: {}, Average MAPE: {:.4f}%".format(MAE / factor, MAPE /factor))
                

    # Log the current train loss
    MAE  = MAE/len(indata.dataset)
    MAPE = MAPE/len(indata.dataset)
    model.history[ridx.trainLoss].append(loss.item())   #get only the loss value
    model.history[ridx.trainMAE].append(MAE)
    model.history[ridx.trainMAPE].append(MAPE)
#--------------------------------------------------------------------------------------
# Start of Train Regressor.

def test_regressor(model, args, device, testLoader, lossFunction = nn.MSELoss()):
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
    print("Commence Testing!")        
    MAE  = 0 
    MAPE = 0
    loss = 0
    testSize = len(testLoader.dataset)
    batchSize = args[1]
    # Inform Pytorch that keeping track of gradients is not required in
    # testing phase.
    with torch.no_grad():
        for data, label in testLoader:
            data = model.shape_input(data)     # select specified columns only.
            data, label = data.to(device), label.to(device)
            pred = model.forward(data).view_as(label)
            # Sum all loss terms and tern then into a numpy number for late use.
            loss = lossFunction(pred, label).item()
            MAE  += torch.FloatTensor.abs(pred.sub(label)).sum().item()
            MAPE += torch.FloatTensor.abs(pred.sub(label)).div(label).mul(100).sum().item()

            

    print("History size {}, idx {}".format(len(model.history), ridx.testLoss))
    # Log the current train loss
    MAE  = MAE/ testSize
    MAPE = MAPE/testSize  
    loss = loss 
    model.history[ridx.testLoss].append(loss)   #get only the loss value
    model.history[ridx.testMAE].append(MAE)
    model.history[ridx.testMAPE].append(MAPE)

    # Print Regressor's evaluation report!
    print("Average MAE: {}, Average MAPE: {:.4f}%, Agv Loss: {:.4f}".format(MAE, MAPE, loss))

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
