# online-learning
A repository for implementing networks and developing learning algorithms, in the online learning scheme.

## Structure

#### **Code**

1. *Solvers*      : This folder contains custom solvers, to be used for finding e-optimal model parameters. They inherit from the optimizer class and implement, init(), state() and step() methods.
2. *Architectures*: This folder contains network architectures, to be used for forecasting, classification etc; they are organizer in folders specific to a problem/ application.
3. *Data*:        : Here lie various datasets. Each one has a detailed desceription.
4. *Datasets*:        : Here go classes that can tranform the raw data files in the 3. Data
folder, into PyTorch datasets. Each such class can transform the raw data as input to a variety of
architectures. 
5. *Tools* :  Here go various utility classes. Main assets: the printing and general training and testing functions, plus anything that is generaly used or not apt for the aove categories.

#### **Applications** 
Here are the top level modules that combine an architecture, a solver and other required pre or post processing steps to efficiently tackkle a problem, such as GEF electrical load prediction. Saved models, results, plots are also stored in the appropriate subfolders.






