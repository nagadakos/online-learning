import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optm
import matplotlib.pyplot as plt
import numpy as np
import sys as sys


# \c def sign(x):
    
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
