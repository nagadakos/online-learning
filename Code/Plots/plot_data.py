import matplotlib.pylab as plt
import pandas as pd

file_location_benchmark = "../../Data/GEF/Load/Task1/L1-benchmark.csv"
file_location_true = "../../Data/GEF/Load/Task1/L1-train.csv"

d = pd.read_csv(file_location_true)

load = d['LOAD']
timestamps = d['TIMESTAMP']

# plt.plot(load)
# plt.show()