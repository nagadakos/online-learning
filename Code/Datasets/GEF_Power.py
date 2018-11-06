"""
    Dataset Definition of GEF_14 power dataset.

    Format:   Column 0             Column 1            Column 2           Column 3-27
              Load Zone    Date (MM-DD-YYYY HH:MM)       Load       25 Weather Station Readings
"""

from __future__ import print_function, division
import os
from os import listdir
from pathlib import Path
from os.path import isdir, join
import re
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
    def __init__(self, csvPath = None, task = "Task 1", toShape = None, transform = None, dataRange = [0, 0], trainMode =1):
        """ Description: This class defines the tranformation of a raw data csv file
                         to a PyTorch dataset. If required it should handle the 
                         complete reshaping and saving of the raw data to an 
                         architecture specific format. 
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
        self.offset = 35064   # labeled data line offset of original GEF file.
        defaultRawFolder = dir_path+"/../../Data/GEF/Load"
        self.descr = "power_GEF_14"
        self.transform = transform
        self.mean = 61.32
        self.std  = 16.71 
        self.min  = 0.0 
        self.max  = 104.0 
        self.maxPower = 315.6
        self.minPower = 48.4
        self.trainMode = trainMode
        self.rawDataFolder = csvPath if csvPath is not None else defaultRawFolder
        self.baseDate= datetime(2005, 1, 1, 1, 00) # labelled data start date
        self.reshaped = 0
        self.toShape = toShape
        self.reshapedBaseFolder = join(dir_path,"../../Applications", self.descr, "ShapedData")
        self.lowerBnd = abs(dataRange[0])
        self.upperBnd = abs(dataRange[1])
        # Dictionary of all the folders in the raw data directory, with all the files
        # per folder, that match the given expression (2nd arg).
        self.rawContents = get_files_from_path(self.rawDataFolder, "*.csv")
        # If raw data has to be reshaped for an architecture, it is done here.
        # for folder, files in contents.items():

        if self.toShape is not None:
            self.reshaped = 1
            self.create_shaped_data(self.rawContents)
            expression = self.descr + "_reshaped_as_" + self.toShape+ "_" +self.transform+".csv"
            self.dataContents =  get_files_from_path(self.reshapedBaseFolder, expression)
            self.activeDataFolder = self.reshapedBaseFolder
        else:
            expression = "*train.csv"
            self.dataContents = get_files_from_path(self.rawDataFolder, expression)  
            self.activeDataFolder = self.rawDataFolder
        # Load the data appropriate for the task; check for bounds validity
        self.data, self.labels = self.load_data(task)
 
        # Get len integer from Range object
        self.data_len = len(self.data.index)
              
    # End of init
    # ---------------------------------------------------------------------------------

    
    def __getitem__(self, index):
        '''Description: mandatory function implementation for dataloaders.

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
            label = np.array(self.labels[index], dtype = float)
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
                    label = (label - self.minPower) / (self.maxPower - self.minPower)

            # Need to reshape label as a nd array of 1,1 to feed to torch.from_numpy   
            itemLabel = np.reshape(label, (1,1))
            itemData = np.concatenate((itemDate, itemData))
        # if the reshaped data file is needed.
        else:
            itemData = np.array(self.data.iloc[index,:], dtype = float)
            itemLabel = np.array(self.labels.iloc[index], dtype = float)
                    # Turn numpy arrays to Tensors for pytorch usage.
        itemData = torch.from_numpy(itemData).float()
        itemLabel = torch.from_numpy(itemLabel).float()

        return (itemData, itemLabel)

    # End of get item
    # ---------------------------------------------------------------------------------

    def __len__(self):
        
        return (self.data_len) # of how many examples(images?) you have

    # -----------------------------------------------------------------------------------------
    # Custom Functions below this point
    # -----------------------------------------------------------------------------------------

    def get_item_size(self):

        return  self.__getitem__(1)[0].size()[0]

    def get_data_descr(self):
        '''
            Get a string describing the clomuns of data.
        '''
        print("0: Trend  1: Month 2-8: day 9: year 10: hour  11-35: data ")

    #------------------------------------------------------------------------------------------
    # Start of shape data 

    def shape_data_function(self, architecture, savePath):
        ''' Description: This function reshapes the read file to the required
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
        month = list(map(lambda x: x.month, date))

        # # Transform day value to 1 hot.
        to1Hot = np.zeros((self.data_len, 7))
        for i, w in enumerate(weekday):
            to1Hot[i, w] = 1

        # -----------------------------------------------------------------------------------
        # Start the Architecture Reshape -specifc code.
        # Remember that class names follow CapWords naming convention. Acronyms should be all
        # caps.
        self.data = self.data.iloc[:, 2:]    # only select load and w1,2 for temperature
                                                  # pandas slice is [lower:upper) 
        temperatures = np.array(self.data.iloc[:, 1:], dtype = float)
        loads = np.array(self.data.iloc[:,0], dtype = float)

        if architecture == "ANNGreek":

            temperatures[:,0] = np.amin(temperatures, axis = 1)
            temperatures[:,1] = np.amax(temperatures, axis = 1)
            temperatures = temperatures[:, :2]

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
                    reshapedData[:, 0:47] /= (self.maxPower - self.minPower)
                    reshapedData[:,55:]   -= self.min
                    reshapedData[:,55:]   /= (self.max - self.min)
                    self.labels -= self.minPower
                    self.labels /= (self.maxPower - self.minPower)
            # Append the labels
            self.labels = self.labels[48:] # first 48 lines are used as data in this architecture
            self.labels = np.reshape(self.labels, (self.labels.shape[0],1))

        #-------------------------------------------------------------------------------------            

        elif architecture == "GLMLF-C2":
            
            # Y = Trend + Month + T_Max_month + T_Max_month^2 + T_Max_Month^3
            temperatures[:,0] = np.amax(temperatures, axis = 1)
            temperatures = temperatures[:, :1]
            #  16 = 1 for Trend, 12 for Month encoding, 1 t_max, 1 T_max^2, 1 T_max^3
            reshapedData = np.zeros((self.data_len, 1+12+1+1+1))
            T_max0 = 77 # Got from the raw data
            j = 0
            # # Transform day value to 1 hot.
            to1HotMonth = np.zeros((self.data_len, 12))
            for i, w in enumerate(month):
                to1HotMonth[i, w-1] = 1 # month ranges from 1 to 12

            for i in range(reshapedData.shape[0]):
                reshapedData[i][0] = trend[i]
                reshapedData[i][1:13] = to1HotMonth[i,:]
                
                # Find monthly max
                if self.trainMode == 1:
                    # curMax = maxMonthlytemperatures[i]
                    curMax = np.amax(temperatures[0:i+1])   # default flatten is row-major
                else:
                    curMax = np.amax(temperatures[0:i+1])   # default flatten is row-major. Remember,
                                                            # python slice is not inclusive of the upper bound
                reshapedData[i][13] = curMax  
                reshapedData[i][14] = curMax * curMax
                reshapedData[i][15] = curMax * curMax * curMax

            if self.transform is not None:
                # Use variable below for normalization to 0-1 
                p1_min   = self.min
                p1_max   = self.max
                p1_denom = p1_max - p1_min
                p2_min   = self.min * self.min
                p2_max   = self.max * self.max
                p2_denom = p2_max - p2_min
                p3_min   = self.min * self.min * self.min
                p3_max   = self.max * self.max * self.max
                p3_denom = p3_max - p3_min

                if self.transform == "normalize":
                    reshapedData[:,13]   -= p1_min
                    reshapedData[:,13]   /= p1_denom
                    reshapedData[:,14]   -= p2_min
                    reshapedData[:,14]   /= p2_denom
                    reshapedData[:,15]   -= p3_min
                    reshapedData[:,15]   /= p3_denom

                    self.labels -= self.minPower
                    self.labels /= (self.maxPower - self.minPower)
            # By this point labels are in the shape (n,), as in all elements in one cell
            # Nee to transform it to (n,1) so it becomes a proper array and can be concatenated
            self.labels = np.reshape(self.labels, (self.labels.shape[0],1))
        print("Len of data {}, len of labels {}".format(reshapedData.shape, self.labels.shape))
        reshapedData = np.concatenate([reshapedData, self.labels], axis = 1)
        print("Reshaped file will be saved at: " + savePath)
        np.savetxt(savePath, reshapedData,  fmt="%.4f", delimiter=",") 
        #-------------------------------------------------------------------------------------            

    # end of shape data 
    # -------------------------------------------------------------------------------------------------
    def comp_stats(self):
        ''' Call to compute mean, max and 1st order moments.
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

    def create_shaped_data(self, contents):
        for folder, files in contents.items():
            offset = self.offset if folder == "Task 1" else 0
            if self.toShape is not None:
                # Form required file
                fileExt = ".csv"

                if self.transform is not None:
                    relativePathBody = join(self.reshapedBaseFolder, folder)
                    if not os.path.isdir(relativePathBody):
                        os.makedirs(relativePathBody, mode=0o777)

                    relativePathBody += "/" + self.descr + "_reshaped_as_"
                    relativePathBody += self.toShape

                    fileExt = "_" + self.transform + ".csv"
                    reshapedPath = relativePathBody + fileExt

                # Indicate that the reshaped input is required
                self.reshaped = 1
                # Check if required reshaped file already exists. Otherwise make it now.
                if not os.path.isfile(reshapedPath):
                   print(" Data file with required input does not exist. Reshaping raw file now.") 
                   fileToRead    = join(self.rawDataFolder ,folder, files[1])
                   self.data     = pd.read_csv(fileToRead, skiprows = offset)
                   self.labels   = np.asarray(self.data.iloc[:, 2]) # third column has the load values
                   self.data_len = len(self.data.index)
                   self.shape_data_function(self.toShape, reshapedPath)

        # print(relativePathBody)
    # End of create shaped data
    # ---------------------------------------------------------------------------------

    def load_data(self, task):

        for folder, files in self.dataContents.items():
            if folder == task:
                fileToRead = join(self.activeDataFolder, folder, files[0]) 
                break
        print(fileToRead)
        # Read into pandas the appropriate data file
        data = pd.read_csv(fileToRead)
        # Check if given read data bounds are valid. If not compensate in the function below.
        self.lowerBnd, self.upperBnd = self.check_bound_validity(len(data.index), task)
        print("LB: {} UP: {}".format(self.lowerBnd, self.upperBnd))
        # Reshaped data had the label, always at the last column.
        # Raw data has the labels at 3 column.
        if self.toShape is not None:
            # labels = np.asarray(data.iloc[self.lowerBnd:self.upperBnd, -1]) # Reshaped data has last column as
            labels = data.iloc[self.lowerBnd:self.upperBnd, -1] # Reshaped data has last column as
            data   = data.iloc[self.lowerBnd:self.upperBnd,0:-1] # Last line is label, take it off our data structure
        else:
            labels  = np.asarray(data.iloc[self.lowerBnd:self.upperBnd, 2]) # third column has the load values
            data   = data.iloc[self.lowerBnd:self.upperBnd, 3:] # Weather data starts at column 3+

        print("Len of dataset: {}".format(len(data.index)))
        return data, labels

    # End of load data
    # ---------------------------------------------------------------------------------

    def check_bound_validity(self, fileSize, task):
        
        dataRange = [self.lowerBnd, self.upperBnd]
        print(dataRange)
        print(fileSize, self.offset)
        # Check for line read range validity. If UB < LB, reverse them, unless UB = 0.
        # UB = 0, means read the whole file from LB to end.
        if dataRange[1] < dataRange[0] :
           if dataRange[1] != 0:
               print("Data Range provided in invalid. Upper bound {} is smaller than Lower bound {}! Will read the file from lower bound to end instead!".format(dataRange[1], dataRange[0]))
               temp = dataRange[0]
               dataRange[0] = dataRange[1]
               dataRange[1] = temp
           else:
               dataRange[1] = fileSize + self.offset if task == "Task 1" else fileSize

        print(dataRange)
        # Only task 1 has file size larger than offset and actually need the offset check
        if task == "Task 1":
            # if raw data does not have to be reshaped, just read it, taking offset to account.
            if self.toShape is None: 
                if dataRange[0] < self.offset:
                    print("Labeled Data starts as line {}. Offsetting accordingly.".format(offset))
                    dataRange[0] = self.offset
            else:
                # New reshaped file does not have the offset unlabeld lines
                # So, we need to account for that.
                dataRange[0] = max(dataRange[0] - self.offset,0)  
                dataRange[1] = max(dataRange[1] - self.offset,0)  

        if dataRange[0] > fileSize:
            print("Lower bound given for data loading is larger than target file. {} vs {}".format(dataRange[0], fileSize))
            print("Setting data reading lower bound to 0")
            dataRange[0] = 0
        if dataRange[1] > fileSize:
            print("Upper bound given for data loading is larger than target file. {} vs {}".format(dataRange[1], fileSize))
            print("Setting data reading upper bound to fileSize: {}".format(fileSize))
            dataRange[1] = fileSize

        return dataRange[0], dataRange[1]
# End of Class
#=======================================================================

def get_files_from_path(targetPath, expression):

    # Find all folders that are note named Solution
    f = [f for f in listdir(targetPath) if (isdir(join(targetPath, f)) and "Solution" not in f)]  
    # initialize a list with as many empty entries as the found folders.
    l = [[] for i in range(len(f))]
    # Create a dictionary to store the folders and whatever files they have
    contents = dict(zip(f,l))
    # print(contents)
    # Pupulate the dictionary with files that match the expression, for each folder.
    for folder, files in contents.items():
        stuff = sorted(Path(join(targetPath, folder)).glob(expression))
        for s in stuff:
            files.append(os.path.split(s)[1] )
        # print(folder, files)
    return contents

#-----------------------------------------------------------------------------------------------------------------------
#  main function definition. Used to debug.
#-----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":

    print(" GEF Power Dataset as Main ")
    myDataset = GefPower( toShape = "ANNGreek", transform = "normalize",
                         dataRange=[0, 76799])

    myDataset2 = GefPower(task = "Task 1", transform = "normalize",
                         dataRange= [76800,0])

    # print(myDataset)
    # myDataset.get_data_descr()
    print(myDataset.__getitem__(1))
    # print(myDataset.__len__())
    # print(myDataset2.__len__())
    # item, label  = myDataset2.__getitem__(3)
    # print(item, label)
    print("Size of an instance {}".format(myDataset.get_item_size()))
