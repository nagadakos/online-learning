import torch
import torchvision
import torchvision.datasets as tdata
import torchvision.transforms as tTrans
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optm
import numpy as np


def train(net, args, device, indata, optim):
        true = 0
        acc  = 0
        for idx, (img, label) in enumerate(indata):
            data, label = img.to(device), label.to(device)
            # forward pass calculate output of model
            output      = net.forward(data)
            pred   = output.max(dim = 1, keepdim = True)
            true  += label.eq(pred[1].view_as(label)).sum().item()
            # print(data.shape)
            # print(output.shape)
            # compute loss
            # loss        = F.nll_loss(output, label)
            loss        = F.cross_entropy(output, label)
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
        acc = true/len(trainLoader.dataset)
        net.history[trainLoss].append(loss)   
        net.history[trainAcc].append(acc)
    
# Testing and error reports are done here
    def test(net, device, testLoader):
        print("Commence Testing!")        
        loss = 0 
        true = 0
        acc  = 0
        # Inform Pytorch that keeping track of gradients is not required in
        # testing phase.
        with torch.no_grad():
            for data, label in testLoader:
                data, label = data.to(device), label.to(device)
                # output = net.forward(data)
                output = net.forward(data)
                # Sum all loss terms and tern then into a numpy number for late use.
                loss  += F.cross_entropy(output, label, reduction = 'elementwise_mean').item()
                # Find the max along a row but maitain the original dimenions.
                # in this case  a 10 -dimensional array.
                pred   = output.max(dim = 1, keepdim = True)
                # Select the indexes of the prediction maxes.
                # Reshape the output vector in the same form of the label one, so they 
                # can be compared directly; from batchsize x 10 to batchsize. Compare
                # predictions with label;  1 indicates equality. Sum the correct ones
                # and turn them to numpy number. In this case the idx of the maximum 
                # prediciton coincides with the label as we are predicting numbers 0-9.
                # So the indx of the max output of the network is essentially the predicted
                # label (number).
                true  += label.eq(pred[1].view_as(label)).sum().item()
        acc = true/len(testLoader.dataset)
        net.history[testAcc].append(acc)
        net.history[testLoss].append(loss/len(testLoader.dataset)) 
        # Print accuracy report!
        print("Accuracy: {} ({} / {})".format(acc, true,
                                              len(testLoader.dataset)))
