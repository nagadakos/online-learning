import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optm
import matplotlib.pyplot as plt
import numpy as np
import sys
import os 


# Add this files directory location, so we can include the index definitions
dir_path = os.path.dirname(os.path.realpath(__file__))

# add this so we can import the trainer and tester modules
tools_path = os.path.join(dir_path, "../")
sys.path.insert(0, tools_path)

from Tools import trainer
import regression_idx as ridx


# End Of imports -----------------------------------------------------------------
# --------------------------------------------------------------------------------


sign = lambda x: ('+', '')[x < 0]

class ANNLFS(nn.Module):
    '''
        Simple single layer feedforwards network, for load preditction.
        It is found in Hond Tao's PhD dissertation as the entry level
        articificial neural netowork architecture.

        Inputs:     (u int) Hourly Linear Trend: Each hour has its owh index, 
                            on the data set. Starting from a base date, each 
                            hour os enumerated. So f dataset starts at 
                            1-1-2001 00:00, hour 00:00 is 0, 01:00 is 1, 02:00 
                            is 2, etc.
                    (float) Temperature for that hour. If many are present in 
                            the dataset, start by using the average.


        Returns:    A single value prediction of the load.
    '''
    history = [[] for i in range(ridx.logSize)]
    def __init__(self, inSize = 2, loss = nn.MSELoss()): 
        super(ANNLFS, self).__init__() 
        self.linear = nn.Linear(inSize, 1)  # 10 nodes are specified in the thesis.
        self.loss = loss
  

    def forward(self, x): 
        x = F.relu(self.linear(x)) 
        return x 

    def get_model_descr(self):
        """Creates a string description of a polynomial."""
        model = 'y = '
        modelSize =len(self.linear.weight.data.view(-1)) 
        for i, w in enumerate(self.linear.weight.data.view(-1)):
                model += '{}{:.2f} x{} '.format(sign(w), w, modelSize - i)
        model += '{:+.2f}'.format(self.linear.bias.data[0])
        return model   

    # Call this function to facilitate the traiing process
    # While there are many ways to go on about calling the
    # traing and testing code, define a function within
    # a "Net" class seems quite intuitive. Many examples
    # do not have a class function; rather they set up
    # the training logic as a block script layed out.
    # Perhaps the object oriented approach leads to less 
    # anguish in large projects...
    def train(self, args, device, indata, optim, lossFunction = nn.MSELoss()):
        true = 0
        acc  = 0
        epochs = args[0]
        
        trainerArgs = args
        column_select = torch.tensor([0, 5])
        for e in range(epochs):
           trainerArgs[0] = e 
           trainer.train_regressor(self, args, device, indata, optim, lossFunction,
                                    column_select)
    
    # Testing and error reports are done here

    def test(self, device, testLoader):
        print("Commence Testing!")        
        loss = 0 
        true = 0
        acc  = 0
        # Inform Pytorch that keeping track of gradients is not required in
        # testing phase.
        with torch.no_grad():
            for data, label in testLoader:
                data, label = data.to(device), label.to(device)
                # output = self.forward(data)
                output = self.forward(data)
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
        self.history[testAcc].append(acc)
        self.history[testLoss].append(loss/len(testLoader.dataset)) 
        # Print accuracy report!
        print("Accuracy: {} ({} / {})".format(acc, true,
                                              len(testLoader.dataset)))

    def report(self):

        print("Current stats of ANNSLF:")
        print("MAE:           {}" .format(self.history[ridx.trainMAE]))
        print("MAPE:          {}" .format(self.history[ridx.trainMAPE]))
        print("Training Loss: {}" .format(self.history[ridx.trainLoss]))
        print("Test MAE:      {}" .format(self.history[ridx.testMAE]))
        print("Test MAPE:     {}" .format(self.history[ridx.testMAE]))
        print("Test Loss:     {}" .format(self.history[ridx.testLoss]))



