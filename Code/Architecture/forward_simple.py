import torch
import torchvision
import torchvision.datasets as tdata
import torchvision.transforms as tTrans
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optm
import matplotlib.pyplot as plt
# Python Imaging Library
import PIL
import numpy as np
import sys as sys

# Index definitions to be used with history log.
trainAcc = 0
trainLoss= 1
testAcc  = 2
testLoss = 3


# Parameters
batch   = 64
epochs  = 100
gamma   = 0.01
momnt   = 0.5
# device  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# # bundle common args to the Dataloader module as a kewword list.
# # pin_memory reserves memory to act as a buffer for cuda memcopy 
# # operations
# comArgs = {'shuffle': True,'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

# # Data Loading -----------------------
# # ******************
# # At this point the data come in Python tuples, a 28x28 image and a label.
# # while the label is a tensor, the image is not; it needs to be converted.  
# # So we need to transform PIL image to tensor and then normalize it.
# # Normalization is quite a good practise to avoid numerical and convergence
# # problems. For that we need the dataset's mean and std which fortunately
# # can be computed!
# # ******************
# mean = 0.1307
# std  = 0.3081
# # Bundle our transforms sequentially, one after another. This is important.
# # Convert images to tensors + normalize
# transform = tTrans.Compose([tTrans.ToTensor(), tTrans.Normalize( (mean,), (std,) )])
# # Load data set
# mnistTrainset = tdata.MNIST(root='../data', train=True, download=True, transform=transform)
# mnistTestset = tdata.MNIST(root='../data', train=False, download=True, transform=transform)

# # Once we have a dataset, torch.utils has a very nice lirary for iterating on that
# # dataset, wit hshuffle AND batch logic. Very usefull in larger datasets.
# trainLoader = torch.utils.data.DataLoader(mnistTrainset, batch_size = batch, **comArgs )
# testLoader = torch.utils.data.DataLoader(mnistTestset, batch_size = 10*batch, **comArgs)
# End of DataLoading -------------------


# Sanity Prints---
# print(len(mnistTrainset))
# print(type(mnist_trainset[0]))

# ----------------


# Model Definition
#-----------------------------
# define network aas a Class
class Net(nn.Module):

    # Class variables for measures.
    trainAcc = 0
    trainLoss= 0
    testAcc  = 0
    testLoss = 0
    # History Log
    # train acc, train loss, test acc, test loss
    history = [[],[],[],[]]
    # Mod init + boiler plate code
    # Skeleton of this network; the blocks to be used.
    # Similar to Fischer prize building blocks!
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64*144, 128)
        self.fc2 = nn.Linear(128, 10)

    # Set the aove defined building blocks as an
    # organized, meaningful architecture here.
    def forward(self, x):
        x = F.relu(self.conv1(x), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 64*144)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x

    # This function will return the number of features
    # required to turn a tensor to a !d Tensor. We exclude batch
    # size. So a [64, 64, 5, 5] tensor will have 1600 features and
    # as such requires a 1, 1600 place holder. Notice we ignore the first
    # 64, as that is the bactc size which is irrelevant for this operation.
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:       # Get the products
            num_features *= s
        return num_features
  
    # Call this function to facilitate the traiing process
    # While there are many ways to go on about calling the
    # traing and testing code, define a function within
    # a "Net" class seems quite intuitive. Many examples
    # do not have a class function; rather they set up
    # the training logic as a block script layed out.
    # Perhaps the object oriented approach leads to less 
    # anguish in large projects...
    def train(self, args, device, indata, optim):
        true = 0
        acc  = 0
        for idx, (img, label) in enumerate(indata):
            data, label = img.to(device), label.to(device)
            # forward pass calculate output of model
            output      = self.forward(data)
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
        acc = true/len(indata.dataset)
        self.history[trainLoss].append(loss)   
        self.history[trainAcc].append(acc)
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

        print("Current stats of MNIST_NET:")
        print("Accuracy:      {}" .format(self.history[trainAcc]))
        print("Training Loss: {}" .format(self.trainLoss))
        print("Test Accuracy: {}" .format(self.testAcc))
        print("Test Loss:     {}" .format(self.testLoss))


# Execution
#-----------------------------
def main():
    print("######### Initiating MNIST N0-DROP Network Training #########\n")

    model = Net().to(device)
    # optim = optm.SGD(model.parameters(), lr=gamma, momentum=momnt)
    optim  = optm.Adam(model.parameters())
    tTotal = 0
    testIters = 1000
    for e in range(epochs):
        print("Epoch: {} start ------------\n".format(e))
        # print("Dev {}".format(device))
        args = e
        model.train(args, device, trainLoader, optim)
        model.test(device, testLoader)

    # Final report
    # model.report()

    with open('PyTorch_no_drop_rep.txt', 'w') as f:
        for i in range(len(model.history[0])):
            f.write("{:.4f} {:.4f} {:4f} {:.4f}\n".format(model.history[0][i], model.history[1][i],
                                                 model.history[2][i], model.history[3][i]))
    for t in range(testIters):
            model.test(device, testLoader)
    with open('PyTorch_no_drop_eval.txt', 'w') as f:
        for i in range(len(model.history[testAcc])):
            f.write("{:.4f} {:.4f} \n".format(model.history[testAcc][i], model.history[testLoss][i]))

# Define behavior if this module is the main executable.
if __name__ == '__main__':
    main()
