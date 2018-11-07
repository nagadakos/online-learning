
import torch
import sys
import os 
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))






def save_tensor(tensor, delimeter = ' ', filePath = None):
    ''' Description: This function saves a tensor to a txt file. It first copies 
                     it to host memory, turn it into a numpy array and dump it 
                     into a txt file. This is faster that a for loop by an order 
                     of magnitude. Original tensor stays in GPU.
        
        Arguments:  tensor (p Tensor): The tensor containing the results
                    to be written. 

                    delimeter (String): A string thatwill separate data in txt
                    
                    filePath (String): Target path to save file
    '''
    a = tensor.cpu()
    a = a.numpy()
    np.savetxt(filePath, a,  fmt="%.4f", delimiter=delimeter)


