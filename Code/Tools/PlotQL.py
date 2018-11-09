import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import torch
from Code.Tools import QLFunction


QLhistoryBT = []
QLhistoryPT = []
Date = ["10/2010", "11/2010", "12/2010", "1/2011", "2/2011", "3/2011", "4/2011", "5/2011",
        "6/2011", "7/2011", "8/2011", "9/2011", "10/2011", "11/2011", "12/2011"]
quantiles = list(np.linspace(0.01, 0.99, 99))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dir_path = os.path.dirname(os.path.realpath(__file__))
base_path = os.path.join(dir_path,"../../Data/GEF/Load/Task ")
loss_func = QLFunction.QuantileLossFunction(quantiles)

for month_n in range(1, 16):
    task_path = base_path + str(month_n)
    #read benmark csv
    benchmark_path = os.path.join(task_path, "L" + str(month_n) + "-benchmark.csv")
    benchmark = pd.read_csv(benchmark_path)
    b_values = torch.from_numpy(np.array(benchmark.values[:, 2:102], dtype=float)).to(device)

    if month_n != 15:
        task_path = base_path + str(month_n+1)
        train_path = os.path.join(task_path, "L" + str(month_n+1) + "-train.csv")
    else :
        task_path = os.path.join(dir_path, "../../Data/GEF/Load/Solution to Task 15")
        train_path = os.path.join(task_path, "solution15_L.csv")
    train = pd.read_csv(train_path)
    t_values = torch.from_numpy(np.array(train.values[:, 2], dtype=float)).to(device)


    loss = loss_func(b_values, t_values)
    QLhistoryBT.append(loss)

for month_n in range(2, 16):
    # task_path = base_path + str(month_n)
    task_path = os.path.join(dir_path, "../../Applications/power_GEF_14/Logs/RNNsome/Predictions/PredResults")

    predict_path = os.path.join(task_path, "0.5-0.7-0.1-Task " + str(month_n) + "-predictions-.txt")
    predict = open(predict_path, 'r')
    p = []
    for values in predict:
        temp = values.split()
        t_for_hour = []
        for value in temp:
            t_for_hour.append(float(value))
        p.append(t_for_hour)
    p_values = torch.from_numpy(np.array(p, dtype=float)).to(device)

    if month_n != 15:
        task_path = base_path + str(month_n)
        train_path = os.path.join(task_path, "L" + str(month_n) + "-train.csv")
    else :
        task_path = os.path.join(dir_path, "../../Data/GEF/Load/Solution to Task 15")
        train_path = os.path.join(task_path, "solution15_L.csv")
    train = pd.read_csv(train_path)
    t_values = torch.from_numpy(np.array(train.values[-(p_values.size(0)):, 2], dtype=float)).to(device)
    loss = loss_func(p_values, t_values)
    QLhistoryPT.append(loss)


plt.plot(QLhistoryBT, 'b*-')
plt.xlabel('Date', fontsize=18)
plt.xticks(range(16), Date)
plt.setp(plt.xticks()[1], rotation=30)
plt.ylabel('Quantile Loss', fontsize=18)
plt.grid()
plt.show()

plt.plot(range(1, 1+len(QLhistoryPT)), QLhistoryPT, 'r*-')
plt.xlabel('Date', fontsize=18)
plt.xticks(range(16), Date)
plt.setp(plt.xticks()[1], rotation=30)
plt.ylabel('Quantile Loss', fontsize=18)
plt.grid()
plt.show()

plt.plot(QLhistoryBT, 'b*-')
plt.plot(range(len(QLhistoryPT)), QLhistoryPT, 'r*-')
plt.xlabel('Date', fontsize=18)
plt.xticks(range(16), Date)
plt.setp(plt.xticks()[1], rotation=30)
plt.ylabel('Quantile Loss', fontsize=18)
plt.legend(['Bencmark and true value', 'predict and true value'])
plt.grid()
plt.show()