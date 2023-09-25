import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import KFold

df = pd.read_csv("data/heightdata.csv")
# x = torch.tensor(df["Age (in months)"], dtype=torch.float32).unsqueeze(1)
# y_true = torch.tensor(df["50th Percentile Length (in centimeters)"], dtype=torch.float32).unsqueeze(1)


def mae(y_true, y_pred):
    return (y_true - y_pred).abs().mean()


def sample_data(indices):
    return df["50th Percentile Length (in centimeters)"].iloc[indices].values


# Task: Train model with 5-fold cross validation
# Data: https://www.cdc.gov/growthcharts/html_charts/lenageinf.htm#males

kf = KFold(n_splits=5)
epochs = 1000

fold_losses = []

for fold, (train_indexes, test_indexes) in enumerate(kf.split(df)):

    # Train model for epochs
    model = nn.Linear(1, 1)
    loss = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()

        y_pred = model(torch.tensor(train_indexes, dtype=torch.float32).unsqueeze(1))
        y_true = torch.tensor(sample_data(train_indexes), dtype=torch.float32).unsqueeze(1)

        #print(y_pred)
        #print(y_true)

        # print(model(x))

        epoch_loss = loss(y_pred, y_true)

        epoch_loss.backward()  # Backwards Propagation
        optimizer.step()

    # Evaluate Accuracy of fold

    print("-------- Fold", fold, "--------")
    print("Weight:", model.weight.data)
    print("Bias:", model.bias.input_data)

    model.eval()

    y_pred = model(torch.tensor(train_indexes, dtype=torch.float32).unsqueeze(1))
    y_true = torch.tensor(sample_data(train_indexes), dtype=torch.float32).unsqueeze(1)

    fold_losses.append(loss(y_pred, y_true))

    print()
    print("MAE: ", mae(y_true, y_pred).item())
    print("Loss: ", loss(y_pred, y_true))
    print()

print("Fold Losses:", fold_losses)
