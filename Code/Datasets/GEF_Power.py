"""
    Dataset Definition of GEF_14 power dataset.

    Format:   Column 0             Column 1            Column 2           Column 3-27
              Load Zone    Date (MM-DD-YYYY HH:MM)       Load       25 Weather Station Readings
"""

from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
# from dateutil.parser import parse
from datetime import datetime,timedelta
from sklearn.preprocessing import LabelBinarizer

dir_path = os.path.dirname(os.path.realpath(__file__))


class GefPower(Dataset):
    def __init__(self, csvPath, toShape = None, transform = None, dataRange = [0, 0]):
       """  
        Args:
            csvPath (string)  : path to csv file
            transform (PyT TF): Pytorch tranform to act on data.

        Returns:
            itemData (tensor.float): A 36D tensor holding temperature readings
                     from 25 weather stations and a 11D tensor containg date information
                      month(1-12) day(0-7, 1-HOT) year(2005+) hour(00:00-23:) trend(as index)  
                      Trend is the number of hourse after the basedate of the dataset.

                     0: Trend  1: Month 2-8: day 9: year 10: hour  11-35: data   
            itemLabel(tensor.float): A 1D tensor containg the dependant load 
                     variable.
       """
       offset = 35064   # labeled data line offset of original GEF file.
       self.transform = transform
       self.mean = 61.32
       self.std  = 16.71 
       self.min  = 0.0 
       self.max  = 104.0 
       self.maxPower = 315.6
       self.minPower = 48.4

       fileToRead = csvPath
       self.baseDate= datetime(2005, 1, 1, 1, 00) # labelled data start date
       self.reshaped = 0

       # If raw data has to be reshaped for an architecture, it is done here.
       if toShape is not None:
           # Form required file
           fileExt = ".csv"
           if self.transform is not None:
               fileExt = "_" + transform + ".csv"
           relativePathBody = "/../../Applications/power_GEF_14/ShapedData/GEF14_power_reshaped_as_ANNGreek"
           reshapedPath = dir_path+ relativePathBody + fileExt
           # Indicate that the reshaped input is required
           self.reshaped = 1
           # Check if required reshaped file already exists. Otherwise make it now.
           if not os.path.isfile(reshapedPath):
               print(" Data file with required input does not exist. Reshaping raw file now.") 
               print(reshapedPath)
               self.data    = pd.read_csv(fileToRead, skiprows = offset)
               self.labels  = np.asarray(self.data.iloc[:, 2]) # third column has the load values
               self.data_len = len(self.data.index)
               self.shape_data(toShape, reshapedPath)
           fileToRead = reshapedPath
           print("Here1"+ fileToRead)
           # New reshaped file does not have the offset unlabeld lines
           # So, we need to account for that.
           dataRange[0] = max(dataRange[0] - offset,0)  
           dataRange[1] = max(dataRange[1] - offset,0)             
       # if raw data does not have to be reshaped, just read it, taking offset to account.
       else:
           if dataRange[0] < offset:
               print("Labeled Data starts as line {}. Ofsetting accordingly.".format(offset))
               dataRange[0] = offset

       # IF bounds dont make sense reverese them
       if dataRange[1] < dataRange[0] :
           # Except if upper bound is zeor then ignore, and read whole file, starting from lower
           # bound.
           if dataRange[1] != 0:
               print("Data Range provided in invalid. Upper bound {} is smaller than Lower bound {}! Will read the file from lower bound to end instead!".format(dataRange[1], dataRange[0]))
               temp = dataRange[0]
               dataRange[0] = dataRange[1]
               dataRange[1] = temp
       print("Here " + fileToRead)
       self.data    = pd.read_csv(fileToRead, skiprows = dataRange[0], nrows = abs(dataRange[1]-
                                                                             dataRange[0]))
       print(self.data.shape) 
       self.labels  = np.asarray(self.data.iloc[:, 2]) # third column has the load values
              # Get len integer from Range object
       self.data_len = len(self.data.index)
              
    # End of init
    # ---------------------------------------------------------------------------------

   
    def comp_stats(self):
        '''

            Call to compute mean, max and 1st order moments.
            Care, it loads the wholre data set to d that, naively.
            mean = 0      max = 104
            mean = 61.327 std = 16.710
        '''
        data = np.array(self.data.iloc[:,3:], dtype=float)
        dmax = np.max(data) 
        dmean = np.mean(data) 
        dmin = np.min(data) 
        dstd = np.std(data) 
        print("Statistics: \n min:{} max: {}\nmean: {} std: {}".format(dmin, dmax, dmean, dstd))
        return(dmin, dmax, dmean, dstd)

    # End of compute statistics
    # ---------------------------------------------------------------------------------

    def __getitem__(self, index):
        '''
            Description: mandatory function implementation for dataloaders.

            Returns:    data, label    
        '''
        
        # If the raw data file is needed.
        if self.reshaped == 0:
            # Get the date of this item. Add number of elapsed
            # hours since basedate.
            date  = self.baseDate + timedelta(hours=index)
            month, day, year, hour = date.month, date.day, date.year, date.hour 
            weekday, trend = date.weekday(), index
            # Transform day value to 1 hot.
            to1Hot = np.zeros((1,7))
            to1Hot[np.arange(1), weekday] = 1

            # Weather data is column 3+
            itemData  = np.array(self.data.iloc[index, 3:], dtype = float)
            itemLabel = np.array(self.labels[index])
            itemDate  = np.array([trend, month, year, hour])
            # All elements of 1-hot representation, which are 7, have to be inserted
            # starting from the original second position. This is what the [2:] array below
            # does
            itemDate  = np.insert(itemDate, [2,2,2,2,2,2,2], to1Hot)

            # Sanity Checks -----------
            # itemData = itemData[3:] 
            # print(itemData)
            # print(itemData.dtype)
            # print(itemLabel)
            # print(itemLabel.dtype)
            # print(itemDate[:])
            # ---|

            # If any necessary transforms are qreuired, they are
            # performed here
            if self.transform:
                if self.transform == "standardize":    # set mean = 0 and std = 1
                    itemData = itemData - self.mean
                    itemData = itemData / self.std
                if self.transform == "normalize":      # scale to 0-1 range
                    itemData = (itemData - self.min) / (self.max - self.min)

            itemData = np.concatenate((itemDate, itemData))
                    # if the reshaped data file is needed.
        else:
            itemData = np.array(self.data.iloc[index,:], dtype = float)
            itemLabel = np.array(self.data.iloc[index,-1], dtype = float)
        # Turn numpy arrays to Tensors for pytorch usage.
        itemData = torch.from_numpy(itemData).float()
        itemLabel = torch.from_numpy(itemLabel).float()

        return (itemData, itemLabel)

    # End of get item
    # ---------------------------------------------------------------------------------

    def __len__(self):
        
        return (self.data_len) # of how many examples(images?) you have

    def get_item_size(self):

        return  self.__getitem__(1)[0].size()[0]

    def get_data_descr(self):
        '''
            Get a string describing the clomuns of data.
        '''
        print("0: Trend  1: Month 2-8: day 9: year 10: hour  11-35: data ")

    #------------------------------------------------------------------------------------------
    # Start of shape data 

    def shape_data(self, architecture, savePath):
        '''
            Description: This function reshapes the read file to the required
                         input of the given architecture. If the architecture is
                         not implemented, a printout will be dmake and the dataset
                         will be read with its normal shape.

            Arguments:  architecture (string): Desired Architecture
            Returns:    In place.
        '''
        # trend = np.fromfunction(map(lambda x: x*1, range(self.data_len)), dtype = int)
        trend = np.arange(0, self.data_len, dtype=np.intc)
        date  = list(map(lambda x: self.baseDate + timedelta(hours=np.asscalar(x)), trend))
        weekday = list(map(lambda x: x.weekday(), date))

        # # Transform day value to 1 hot.
        to1Hot = np.zeros((self.data_len, 7))
        for i, w in enumerate(weekday):
            to1Hot[i, w] = 1

        # -----------------------------------------------------------------------------------
        # Start the Architecture Reshape -specifc code.
        if architecture == "ANNGreek":
            self.data = self.data.iloc[:, 2:]    # only select load and w1,2 for temperature
                                                  # pandas slice is [lower:upper)  
            temperatures = np.array(self.data.iloc[:, 1:], dtype = float)
            temperatures[:,0] = np.amin(temperatures, axis = 1)
            temperatures[:,1] = np.amax(temperatures, axis = 1)
            temperatures = temperatures[:, :2]
            loads = np.array(self.data.iloc[:,0], dtype = float)

            #  59 = 2 *24 power readings + 7 for the days encoding + 2* (min + max) temperature values 
            reshapedData = np.zeros((self.data_len-48, 48+7+4))
            j = 0
            for i in range(reshapedData.shape[0]):
                reshapedData[i][0:47] = loads[j:j+47]
                reshapedData[i][48:55] = to1Hot[j,:]
                reshapedData[i][55:] = temperatures[j:j+2][:].flatten()  # default flatten is row-major
                j += 1 
            if self.transform is not None:
                if self.transform == "normalize":
                    reshapedData[:, 0:47] -= self.minPower
                    reshapedData[:, 0:47] /= self.maxPower
                    reshapedData[:,55:]   -= self.min
                    reshapedData[:,55:]   /= self.max
                    self.labels -= self.minPower
                    self.labels /= self.maxPower
            # Append the labels
            self.labels = self.labels[48:] # first 48 lines are used as data in this architecture
            self.labels = np.reshape(self.labels, (self.labels.shape[0],1))
            reshapedData = np.concatenate([reshapedData, self.labels], axis = 1)
            np.savetxt(savePath, reshapedData,  fmt="%.4f", delimiter=",") 
        #-------------------------------------------------------------------------------------            
   # end of shape data 
   # -------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
#  main function definition. Used to debug.
#-----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":

    print(" GEF Power Dataset as Main ")
    myDataset = GefPower('../../Data/GEF/Load/Task 1/L1-train.csv', toShape = "ANNGreek", transform = "normalize",
                         dataRange=[0, 76799])

    myDataset2 = GefPower('../../Data/GEF/Load/Task 1/L1-train.csv', transform = "normalize",
                         dataRange= [76800,0])

    print(myDataset)
    myDataset.get_data_descr()
    print(myDataset.__getitem__(1))
    print(myDataset.__len__())
    print(myDataset2.__len__())
    item, label  = myDataset2.__getitem__(3)
    print(item)
    print("Size of an instance {}".format(myDataset.get_item_size()))
