
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os 
import numpy as np
import copy
from pathlib import Path
from os.path import isdir, join, isfile
from os import listdir
import fnmatch

dir_path = os.path.dirname(os.path.realpath(__file__))


class QuantileLoss(nn.Module):
    '''
        Description: This module models Quantile loss. it is implmented as a nn module
                     So as to enable backproagation to compute gradients efficiently.

        Returns:    x (loss): Averaged quantile loss over all input elements.
    '''
    def __init__(self,q):
        super(QuantileLoss, self).__init__()
        if isinstance(q, list):
            self.q = q
        else:
            self.q = [q]
        self.descr = "QuantileLoss_" + str(len(self.q))

    def forward(self, x, target):
        loss = [] # place holder for each quantile loss.
        for i,q in enumerate(self.q): 
            # Each Element of the list, is the quantile loss of each quantile with each output node.
            # So quantile 1 get multipled with output node 1, for each sample.
            # The end results is the element at list[i] is of size: (n, 1).
            x_col = torch.reshape(x[:,i], (len(x),1))
            loss.append( (q* F.relu(target- x_col) + (1-q) * F.relu(x_col - target)).sum(dim=1, keepdim=True))

        # Convert List quantile loss Tensors to Tensor.
        # list is len (NumOfQuantiles) and items are(n,1). It becomes a tensor of
        # shape (n, numOfQuantiles)
        loss = torch.cat(loss, dim =1)
        # Compute Mean of all Quantile loss for the whole batch. Returned loss must
        # be scalar for autograd.
        meanLoss = loss.mean()
        return meanLoss, loss

# End of Quantile Loss
# ---------------------------------------------------------------------------------
def compute_loss(filesPath = '', inReps = []):

    # Argument Handler
    # ----------------------
    # This section checks and sanitized input arguments.
    if not filesPath and  not inReps:
        print('No input log path or history lists are given to plot_regressor!!')
        print('Abort plotting.')
        return -1

    if not isinstance(filesPath, list):
        files = [filesPath]
    else:
        files = filesPath
    reps = []

    if filesPath:
        for i,f in enumerate(files):
            reps.append([[] for i in range(ridx.logSize)])
            # print(i)
            # print("Size of reps list: {} {}".format(len(reps),len(reps[i])))
            with open(f, 'r') as p:
                # print("i is {}".format(i))
                for j,l in enumerate(p):
                    # Ignore last character from line parser as it is just the '/n' char.
                    report = l[:-2].split(' ')
                    print(report)
                    # reps[i][ridx.trainMAE].append(report[ridx.trainMAE])
                    # reps[i][ridx.trainMAPE].append(report[ridx.trainMAPE])
                    # reps[i][ridx.trainLoss].append(report[ridx.trainLoss])
                    # reps[i][ridx.testMAE].append(report[ridx.testMAE])
                    # reps[i][ridx.testMAPE].append(report[ridx.testMAPE])
                    # reps[i][ridx.testLoss].append(report[ridx.testLoss])


#
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

def save_model_dict(model, filePath = None):

    torch.save(model.state_dict(), filePath)

def load_model_dict(model, filePath = None):

    model.load_state_dict(torch.load(filePath))

def instantiate_model_from_template(modelTemplate):

    # TODO: Find a way togenerally copy a model from a template, without having to 
    # acutally difine a copy method in each architecture.
    print("Not yet Implemented")

# ------------------------------------------------------------------------------------------------------------    
def get_files_from_path(targetPath, expression):

    # Find all folders that are not named Solution.
    d = [f for f in listdir(targetPath) if (isdir(join(targetPath, f)) and "Solution" not in f)]  
    # Find all file in target directory that match expression
    f = [f for f in listdir(targetPath) if (isfile(join(targetPath, f)) and fnmatch.fnmatch(f,
                                                                                            expression))]  
    # initialize a list with as many empty entries as the found folders.
    l = [[] for i in range(len(d))]
    # Create a dictionary to store the folders and whatever files they have
    contents = dict(zip(d,l))
    contents['files'] = f

    # Pupulate the dictionary with files that match the expression, for each folder.
    # This will consider all subdirectories of target directory and populate them with
    # files that match the expression.
    for folder, files in contents.items():
        stuff = sorted(Path(join(targetPath, folder)).glob(expression))
        for s in stuff:
            files.append(os.path.split(s)[1] )
        # print(folder, files)
    for files in contents['files']:
        stuff = sorted(Path(join(targetPath, files)).glob(expression))
    # print(contents)
    return contents


# ----------------------------------------------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------------------------------------------
def main():
    # add this so we can import the trainer and tester modules
    # archPath = os.path.join(dir_path, "../")
    # sys.path.insert(0, archPath)
    # from Architecture import ann_greek

    # model = ann_greek.ANNGreek(59, 99, "SGD", lr=0.5,momnt=0.7, wDecay=0.1).to('cuda:0')
    # filePath ='../../Applications/power_GEF_14/Models/ANNGreek/ANNGreek-ANNGreek-SGD-lr-0.5-momnt-0.7-0.1-SGD-trainedFor-2' 
    # load_model_dict(model, filePath)
    # print(model)
    # # parameters() need to be put in a list to be printed
    # print(list(model.parameters()))

    filePath = "../../Data/GEF/Load/"
    f = get_files_from_path(filePath, "*benchmark.csv")

    print(f)
    files = []
    for i in f['files']:
        files.append(join(filePath, i)) 
    print(files)




if __name__ == '__main__':
    main()
