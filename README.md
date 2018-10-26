# online-learning
A repository for implementing networks and developing learning algorithms, in the online learning scheme.

## Structure

#### **Code**

1. *Solvers*      : This folder contains custom solvers, to be used for finding e-optimal model parameters. They inherit from the optimizer class and implement, init(), state() and step() methods.
2. *Architectures*: This folder contains network architectures, to be used for forecasting, classification etc; they are organizer in folders specific to a problem/ application.
4. *Datasets*:        : Here go classes that can tranform the raw data files in the 3. Data
folder, into PyTorch datasets. Each such class can transform the raw data as input to a variety of
architectures. 
5. *Tools* :  Here go various utility classes. Main assets: the printing and general training and testing functions, plus anything that is generaly used or not apt for the aove categories.

#### **Applications** 
Here are the top level modules that combine an architecture, a solver and other required pre or post processing steps to efficiently tackkle a problem, such as GEF electrical load prediction. Saved models, results, plots are also stored in the appropriate subfolders.

There is a folder for each broad task at hand. For example, to predict load values with the GEF 14 data set, execute the top_level.py file found in the appropriate folder.
Each top_level for each problem, has a variety of architectures implemented, simply pass as an argument which one is desired. For a list of each of the implemented arcitectures and a gui on how to implement a new one refer to each of the readme files in the appropriate application folder.


#### **Data**    
Here lie various raw  data files. Each one has a detailed desceription. The need to have and appropriate dataset class defined in the dataset folder, in order to be properly loaded as pytorch datasets.


## Usage

Call the top level module for a task found in the Applications directory, like so:

$python .../Applications/desiredApplication/top_level.py
