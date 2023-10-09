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

def get_unique_items(data):

    items = set()
    for item in data:
        items.add(item)

    return items

# https://stackoverflow.com/questions/71146270/one-hot-encoding-text-data-in-pytorch
def build_one_hot(vocab, keys):
    if isinstance(keys, str):
        keys = [keys]
    return nn.functional.one_hot(torch.tensor(vocab.forward(keys)), num_classes=len(vocab))


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

nlp = spacy.load("en_core_web_sm")

"""
def read_name_tokens():
    output = set()

    for name in processed_data["Name"]:

        doc = nlp(name)
        for token in doc:
            output.add(token.text)

    return output

name_token_data = read_csv_cached("name_tokens.csv", read_name_tokens, True)

name_vocab = torchtext.vocab.build_vocab_from_iterator(list(name_token_data), specials=["<unk>"])

for name in processed_data["Name"].values:
    doc = nlp(name)

    for token in doc:

        print(name_vocab.forward([token.text]))
"""


"""
genre_tokens = list(get_unique_items(df["Genre"].dropna().values))
genre_vocab = torchtext.vocab.build_vocab_from_iterator([genre_tokens], specials=["<unk>"])
genre_one_hot = build_one_hot(genre_vocab, list(processed_data["Genre"].values))

publisher_tokens = list(get_unique_items(df["Publisher"].dropna().values))
publisher_vocab = torchtext.vocab.build_vocab_from_iterator([publisher_tokens], specials=["<unk>"])
publisher_one_hot = build_one_hot(publisher_vocab, list(processed_data["Publisher"].values))
"""

data_true = processed_data["Global_Sales"]
numeric_data = processed_data[["Critic_Score", "Critic_Count", "User_Score"]]
one_hot_dummies_data = pandas.get_dummies(processed_data[["Platform", "Genre"]], prefix=["Platform", "Genre"])

input_data = torch.cat([torch.tensor(one_hot_dummies_data.values), torch.tensor(numeric_data.values)], 1)
print("Input sample:", input_data[0])

kf = KFold(n_splits=5, shuffle=True)
epochs = 2000

fold_losses = []

print("Start Training")
for fold, (train_indexes, test_indexes) in enumerate(kf.split(numeric_data)):

    model = nn.Linear(input_data.shape[1], 1)
    loss = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()

        epoch_input = torch.index_select(input_data, 0, torch.tensor(train_indexes, dtype=torch.int32)).type(torch.float32)


        # https://pytorch.org/docs/stable/generated/torch.reshape.html#torch.reshape
        y_pred = model(epoch_input)
        y_pred = torch.reshape(y_pred, (-1, ))

        y_true = torch.tensor(data_true.iloc[train_indexes].values, dtype=torch.float32)

        epoch_loss = loss(y_pred, y_true)


        epoch_loss.backward()  # Backwards Propagation
        optimizer.step()

    print("-------- Fold", fold, "--------")
    model.eval()

    test_input = torch.index_select(input_data, 0, torch.tensor(test_indexes, dtype=torch.int32)).type(torch.float32)
    y_pred = model(test_input)
    y_pred = torch.reshape(y_pred, (-1,))

    y_true = torch.tensor(data_true.iloc[test_indexes].values, dtype=torch.float32)

    test_loss = loss(y_pred, y_true)
    fold_losses.append(test_loss)
    print("Loss:", test_loss)
    print()

print("------------------------")
print("Average Fold Loss:", torch.tensor(fold_losses).mean())