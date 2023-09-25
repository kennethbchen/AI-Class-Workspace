import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold


def line(x):
    return 3 * x + 4


"""
x = torch.arange(1, 10, 0.1)
print(line(x))

print(x.shape)

x2 = torch.reshape(x, (10, 9))
print(line(x2))
print(x2.shape)

import matplotlib.pyplot as plt

plt.plot(x, line(x))
plt.show()
"""

# https://www.cdc.gov/growthcharts/html_charts/lenageinf.htm#males

df = pd.read_csv("data/heightdata.csv")

# x = df["Age (in months)"]
# y_true = df["50th Percentile Length (in centimeters)"]


def mae(y_true, y_pred):
    return (y_true - y_pred).abs().mean()


"""
#y_pred = line(x)

# Create groups for k-fold cross validation
group_count = 5

size_per_group = int(len(df) / group_count)
print("Total Rows:", len(df), "| Size Per Group:", size_per_group)

shuffled = df.sample(frac=1)

# Split data into k groups
groups = np.array_split(shuffled, group_count)

group_errors = []
error_sum = 0

for test_index in range(group_count):

    error = mae(groups[test_index]["50th Percentile Length (in centimeters)"], line(groups[test_index]["Age (in months)"]))

    error_sum += error
    group_errors.append(error)

print(group_errors)
print("Average error across all groups:", error_sum / len(group_errors))
"""

# ---------


kf = KFold(n_splits=5)

# Task: Perform 5-fold cross validation on some random linear function (no training)

fold_errors = torch.empty(0)
for i, (train_index, test_index) in enumerate(kf.split(df)):
    errors = 0

    test_true = torch.tensor(df.iloc[test_index]["50th Percentile Length (in centimeters)"].values)
    test_pred = line(torch.tensor(test_index))

    fold_errors = torch.cat( (fold_errors, mae(test_true, test_pred).reshape(1)), 0)

print("Error for each fold:", fold_errors)
print("Average error across all folds:", fold_errors.mean())
