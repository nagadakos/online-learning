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
# CapWords naming convention for class names
# If there is a n acronym such as ANN, it is all uppercase
class ANNGreek(nn.Module):
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
        super(ANNGreek, self).__init__() 
        self.firstPass = 1
        self.linear = nn.Linear(inSize, 1)  # 10 nodes are specified in the thesis.
        self.loss = loss
        self.descr = "ANNGreek" 

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

    def shape_input(self, x, label):
        '''
            Shape unput data x to: 
                0-23 : Hourly load of last day
                24-27: Max and min temp of 2 weather station
                30-33: Bit encoding for day
        '''
         
        return x

    def save_history(self, filePath):
        trainer.save_log(filePath, self.history)

    def train(self, args, device, trainLoader, testLoader, optim, lossFunction = nn.MSELoss()):
        epochs = args[0]
        
        trainerArgs = args
        testerArgs = args
        testerArgs[1] *= 4 

        for e in range(epochs):
           trainerArgs[0] = e 
           testerArgs[0] = e 
           trainer.train_regressor(self, args, device, trainLoader, optim, lossFunction)
           self.test(testerArgs, device, testLoader, lossFunction)
    
    # Testing and error reports are done here
    def test(self, args, device, testLoader, lossFunction = nn.MSELoss()):
        testArgs = args
        trainer.test_regressor(self, args, device, testLoader, lossFunction) 


    def report(self):

        print("Current stats of ANNSLF:")
        print("MAE:           {}" .format(self.history[ridx.trainMAE]))
        print("MAPE:          {}" .format(self.history[ridx.trainMAPE]))
        print("Training Loss: {}" .format(self.history[ridx.trainLoss]))
        print("Test MAE:      {}" .format(self.history[ridx.testMAE]))
        print("Test MAPE:     {}" .format(self.history[ridx.testMAE]))
        print("Test Loss:     {}" .format(self.history[ridx.testLoss]))



