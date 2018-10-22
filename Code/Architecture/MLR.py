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

from Tools import trainer

# End Of imports -----------------------------------------------------------------
# --------------------------------------------------------------------------------

sign = lambda x: ('+', '')[x < 0]


class LinearRegression(nn.Module):
    '''
        Simple Linear regression, that is also linear in feature space


        Returns:    A single value prediction
    '''
    history = [[] for i in range(ridx.logSize)]
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

    def shape_input(self, data):

        data = data[:, 11:]
        return data
    def save_history(self, filePath):
        trainer.save_log(filePath, self.history)

    def train(self, args, device, trainLoader, testLoader, optim, lossFunction = nn.MSELoss()):
        true = 0
        acc  = 0
        epochs = args[0]
        
        trainerArgs = args
        testerArgs = args
        testerArgs[1] *= 4 
        # column_select = torch.tensor([i for i in range(5,30)])
        for e in range(epochs):
           trainerArgs[0] = e 
           testerArgs[0] = e 
           trainer.train_regressor(self, args, device, trainLoader, optim, lossFunction)
           self.test(testerArgs, device, testLoader, lossFunction)
    
    # Testing and error reports are done here
    def test(self, args, device, testLoader, lossFunction = nn.MSELoss()):
        print("--Epoch {} Testing ----" . format(args[0]))        
        testArgs = args
        trainer.test_regressor(self, args, device, testLoader, lossFunction) 



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
