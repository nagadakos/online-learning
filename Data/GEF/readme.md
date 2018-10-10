Every folder in GEF contains a separate data set that can be used for online learning.
Each data set follows the same format: 15 folders names Task 1-15, containing a training and benchmark csv file.
Training files contain data entries, structured hourly value reports, also marked by month. Year ranges vary according to the dataset.
Benchmarks were generated to indicate valid forecasting results. As these sets were used for learning competitions, authors of data-set, suggest to compare results with the top contanders presented in the clsx file, for each data set and task.

A short list of the data-sets:

   Set          Predictor Variables(num)        Target 
1. Load :            w 1-25 (25)                 Price
2. Price:       Zonal Load, Total Load(2)        Price
3. Solar:         12 Variables (12)              Power 
4. Wind*:        weather forecasts 2*u,2*v,      Power
                 old power ratings (4 or 5)

* Has seperate files for each of the farms(10). Ratings of u (zonal) and v (meridional) are given in 2 seperate altitudes: 10m, 100m.




Source of dataset:

Tao Hong, Pierre Pinson, Shu Fan, Hamidreza Zareipour, Alberto Troccoli and Rob J. Hyndman, "Probabilistic energy forecasting: Global Energy Forecasting Competition 2014 and beyond", International Journal of Forecasting, 2016
[As of Jan, 2016, the paper is in press. Please check ScienceDirect for the most recent status and full citation to the paper.]
