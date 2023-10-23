import numpy as np
import pandas
import torch
import torch.nn as nn
import torchtext
import pandas as pd
from sklearn.model_selection import KFold
import spacy
import os.path
import csv
import codecs
from util import read_csv_cached
from gensim.models import KeyedVectors

"""
    Data Processing On CSV: Remove "tbd" from User_Score / all rows
"""

df = pd.read_csv("data/video_games_sales.csv")
df["User_Score"] = df["User_Score"].astype(float)
df["User_Count"] = df["User_Count"].astype(float)

"""
Name: Tokenize?
Platform: 
Year_of_Release
Genre: Tokenize
Publisher: Tokenize

Critic_Score
Critic_Count
User_Score
User_Count

Developer Rating

Predict: Global_Sales

"""


processed_data = df.dropna()

def read_word_embeddings():
    print("Loading word-embedding related data")
    nlp = spacy.load("en_core_web_sm")
    word_vectors = KeyedVectors.load_word2vec_format("ignore/GoogleNews-vectors-negative300.bin", binary=True)

    name_embeddings = []

    print("Creating name embeddings")
    max_len = 3
    for name in processed_data["Name"].values:
        doc = nlp(name)
        embeddings = []


        for token in doc:
            if token.text in word_vectors:
                embeddings.append(word_vectors[token.text])
            else:
                # Not found, pad zeroes
                embeddings.append(np.zeros((300,)))

        # Resize embeddings if needed
        if len(embeddings) < max_len:
            padding = [np.zeros((300,)) for _ in range(max_len - len(embeddings))]
            embeddings = padding + embeddings
        elif len(embeddings) > max_len:
            embeddings = embeddings[:max_len]

        # Flatten embeddings
        embeddings = [item for sublist in embeddings for item in sublist]
        name_embeddings.append(embeddings)

    return name_embeddings

# Checks if GPU can be used
# https://pytorch.org/get-started/locally/
if torch.cuda.is_available():
    print("Cuda is available")
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


data_true = processed_data["Global_Sales"]

numeric_data = processed_data[["Critic_Score", "Critic_Count", "User_Score", "User_Count"]]
# Normalize
numeric_data = ( (numeric_data - numeric_data.mean()) / numeric_data.std() )

one_hot_dummies_data = pandas.get_dummies(processed_data[["Platform", "Genre"]], prefix=["Platform", "Genre"])
name_embeddings = np.array(read_csv_cached("ignore/cached_data/name_embeddings.csv", read_word_embeddings)).astype(float)

input_data = torch.cat([torch.tensor(name_embeddings), torch.tensor(one_hot_dummies_data.values), torch.tensor(numeric_data.values)], 1).to(device)
input_size = input_data.shape[1]

print("Dataset size:", len(input_data), "rows")
print("Input size:", input_size, "values")
print("Input sample:", input_data[0])

kf = KFold(n_splits=5, shuffle=True)
epochs = 1000

fold_losses = []

print("Start Training")
for fold, (train_indexes, test_indexes) in enumerate(kf.split(numeric_data)):

    model = nn.Sequential(
        nn.Linear(input_size, 5), # Input Layer
        nn.ReLU(),
        nn.Linear(5, 1), # Output Layer
        nn.ReLU()
    ).to(device)

    loss = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.009)

    epoch_input = torch.index_select(input_data, 0, torch.tensor(train_indexes, dtype=torch.int32).to(device)).type(
        torch.float32)

    test_input = torch.index_select(input_data, 0, torch.tensor(test_indexes, dtype=torch.int32).to(device)).type(
        torch.float32).to(device)


    for epoch in range(epochs):
        model.train()

        optimizer.zero_grad()

        # https://pytorch.org/docs/stable/generated/torch.reshape.html#torch.reshape
        train_pred = model(epoch_input)
        train_pred = torch.reshape(train_pred, (-1, ))

        train_true = torch.tensor(data_true.iloc[train_indexes].values, dtype=torch.float32).to(device)

        epoch_loss = loss(train_pred, train_true)
        epoch_loss.backward()  # Backwards Propagation
        optimizer.step()

        # Evaluate epoch
        model.eval()

        test_pred = model(test_input)
        test_pred = torch.reshape(test_pred, (-1,))

        test_true = torch.tensor(data_true.iloc[test_indexes].values, dtype=torch.float32).to(device)

        test_loss = loss(test_pred, test_true)
        fold_losses.append(test_loss)

        print("Fold", fold, "Epoch", epoch, "Train Loss:", epoch_loss.item(), "Test Loss:", test_loss.item())


print("------------------------")
print("Average Fold Loss:", torch.tensor(fold_losses).mean())