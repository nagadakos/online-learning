
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optm
import sys
import os 

dir_path = os.path.dirname(os.path.realpath(__file__))

sys.path.insert(0, dir_path)

import indexes
# Call this function to facilitate the traiing process
# While there are many ways to go on about calling the
# traing and testing code, define a function within
# a "Net" class seems quite intuitive. Many examples
# do not have a class function; rather they set up
# the training logic as a block script layed out.
# Perhaps the object oriented approach leads to less 
# anguish in large projects...
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


def train_regressor(model, args, device, indata, optim, lossFunction = nn.MSELoss()):
    true = 0
    acc  = 0
    for idx, (img, label, date) in enumerate(indata):
        # if idx == 1:
        data, label = img.to(device), label.to(device)
        # forward pass calculate output of model
        output = model.forward(data).view_as(label)
        pred   = output

        # Sanity prints
        # print("Prediction shape: {} label shape: {}". format(pred.shape, label.shape))
        # print("Prediction: {}, labels: {}" .format(pred, label))

        # compute loss
        loss        = lossFunction(pred, label)
        # Backpropagation part
        # 1. Zero out Grads
        optim.zero_grad()
        # 2. Perform the backpropagation based on loss
        loss.backward()            
        # 3. Update weights 
        optim.step()
       # Training Progress report for sanity purposes! 
        if idx % 20 == 0: 
            print("Epoch: {}->Batch: {} / {}. Loss = {}".format(args, idx, len(indata), loss.item() ))
            

    # Log the current train loss
    acc = true/len(indata.dataset)
    model.history[indexes.trainLoss].append(loss.item())   #get only the loss value
    model.history[indexes.trainAcc].append(acc)
