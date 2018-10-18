import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optm
import matplotlib.pyplot as plt
import numpy as np
import sys as sys
import os
# Add this files directory location, so we can include the index definitions
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path)

import regression_idx as ridx

# add this so we can import the trainer and tester modules
tools_path = os.path.join(dir_path, "../")
sys.path.insert(0, tools_path)

from Tools import trainer, tester

# End Of imports -----------------------------------------------------------------
# --------------------------------------------------------------------------------

sign = lambda x: ('+', '')[x < 0]


class LinearRegression(nn.Module):
    '''
        Simple Linear regression, that is also linear in feature space


        Returns:    A single value prediction
    '''
    history = [[], [], [], []]
    def __init__(self, inSize = 1, loss = nn.MSELoss()): 
        super(LinearRegression, self).__init__() 
        self.linear = nn.Linear(inSize, 1)  # Specify inputSize at constructor
        self.loss = loss
  

    def forward(self, x): 
        x = self.linear(x) 
        return x 

    def get_model_descr(self):
        """Creates a string description of a polynomial."""
        model = 'y = '
        modelSize =len(self.linear.weight.data.view(-1)) 
        for i, w in enumerate(self.linear.weight.data.view(-1)):
                model += '{}{:.2f} x{} '.format(sign(w), w, modelSize - i)
        model += '{:+.2f}'.format(self.linear.bias.data[0])
        return model   


    def train(self, args, device, indata, optim, lossFunction = nn.MSELoss()):
        true = 0
        acc  = 0
        epochs = args[0]
        
        trainerArgs = args
        column_select = torch.tensor([i for i in range(5,30)])
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
        print("Accuracy:      {}" .format(self.history[trainAcc]))
        print("Training Loss: {}" .format(self.trainLoss))
        print("Test Accuracy: {}" .format(self.testAcc))
        print("Test Loss:     {}" .format(self.testLoss))
class PolynomialRegression(nn.Module):
    '''
        Linear Regression that accepts polynomial features as input
        
        Note: Not really necessary, as LinearRegression can also handle
                polynomials, if alr4eady provided as input. Maybe here
                we can have a file that defines the polynomial to
                differentiate between the 2.

        Returns:    A single value prediction
    '''

    def __init__(self, inSize = 1): 
        super(PolynomialRegression, self).__init__() 
        self.linear = torch.nn.Linear(inSize, 1)  # Specify inputSize at constructor
  

    def forward(self, x): 
        x = self.linear(x) 
        return x 












def main():

    print("MLR module called as main.")


if __name__ == "__main":
    main()
