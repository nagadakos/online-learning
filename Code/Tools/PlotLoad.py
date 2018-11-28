import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import torch
from Code.Tools import QLFunction

Task_num = 3

quantiles = [0.01*i for i in range(1,100)]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dir_path = os.path.dirname(os.path.realpath(__file__))
base_path = os.path.join(dir_path,"../../Data/GEF/Load/Task ")
loss_func = QLFunction.QuantileLossFunction(quantiles)

task_path = base_path + str(Task_num)
train_path = os.path.join(task_path, "L" + str(Task_num) + "-train.csv")

train = pd.read_csv(train_path)
t1_values = torch.from_numpy(np.array(train.values[:, 2], dtype=float)).to(device)

task_path = base_path + str(Task_num+1)
train_path = os.path.join(task_path, "L" + str(Task_num+1) + "-train.csv")

train = pd.read_csv(train_path)
t2_values = torch.from_numpy(np.array(train.values[:, 2], dtype=float)).to(device)

# Predicted_path = base_path + str(Task_num)
# Predicted_path = os.path.join(Predicted_path, "L" + str(Task_num) + "-benchmark.csv")
# predict = pd.read_csv(Predicted_path)
# predict_path = base_path + str(Task_num+1)
predict_path = os.path.join(dir_path, "../../Applications/power_GEF_14/Logs/RNNsome/Predictions/PredResults")
predict_path = os.path.join(predict_path, "0.01-0.7-0.1-Task " + str(Task_num+1) + "-predictions-.txt")
# predict_path = os.path.join(predict_path, "0.01-0.5-0.5-Task " + str(Task_num+1) + "-predictions-.txt")
predict = open(predict_path, 'r')
p = []
for values in predict:
    temp = values.split()
    t_for_hour = []
    for value in temp:
        t_for_hour.append(float(value))
    p.append(t_for_hour)
predict = torch.from_numpy(np.array(p, dtype=float)).to(device)

p50_values = torch.from_numpy(np.array(predict[:, 49], dtype=float)).to(device)
p90_values = torch.from_numpy(np.array(predict[:, 89], dtype=float)).to(device)

y_train = np.append(t1_values.cpu().numpy(), t2_values.cpu().numpy())
y_p50 = p50_values.cpu().numpy()
y_p90 = p90_values.cpu().numpy()

print(len(y_train))
print(len(y_p50))

plt.plot(range(len(y_train) - len(y_p50), len(y_train)), y_p50, 'r-')
plt.plot(range(len(y_train) - len(y_p90), len(y_train)), y_p90, 'g-')
# plt.fill_between(range(len(y_train) - len(y_p90), len(y_train)), y_p50, y_p90, color='r', alpha=0.7)
plt.plot(y_train, 'b-', alpha=0.7)
plt.vlines(len(t1_values), 0, 350, colors='k', linestyles='solid')
plt.xticks([])
plt.ylabel('Load Value', fontsize=18)
plt.grid()
plt.show()