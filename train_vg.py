import torch
import torch.nn as nn
import torchtext
import pandas as pd
from sklearn.model_selection import KFold

"""
a = torch.tensor([1, 2, 3, 4])
b = torch.tensor([5, 6, 7, 8])
print(torch.cat([a, b]))
exit()
"""

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

genre_tokens = list(get_unique_items(df["Genre"].dropna().values))
genre_vocab = torchtext.vocab.build_vocab_from_iterator([genre_tokens], specials=["<unk>"])

#genre_one_hot = nn.functional.one_hot(torch.tensor(genre_vocab.forward(genre_tokens)), num_classes=len(genre_vocab))

# TODO: get genre_one_hot in with the rest of the numeric_data somehow
#genre_one_hot = build_one_hot(genre_vocab, list(processed_data["Genre"].values))


data_true = processed_data["Global_Sales"]
numeric_data = processed_data[["Critic_Score", "Critic_Count", "User_Score"]]

#print(torch.index_select(genre_one_hot, 0, torch.tensor([1, 2, 3, 4], dtype=torch.int32)))

kf = KFold(n_splits=5, shuffle=False)
epochs = 1000

#print(torch.nn.functional.one_hot(torch.tensor(df["Genre"].values), get_class_count(df["Genre"].values)))

print("Start Training")
for fold, (train_indexes, test_indexes) in enumerate(kf.split(numeric_data)):

    model = nn.Linear(3, 1)
    loss = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()


        # https://pytorch.org/docs/stable/generated/torch.reshape.html#torch.reshape
        y_pred = model(torch.tensor(numeric_data.iloc[train_indexes].values, dtype=torch.float32).unsqueeze(1))
        y_pred = torch.reshape(y_pred, (-1, ))

        y_true = torch.tensor(data_true.iloc[train_indexes].values, dtype=torch.float32)

        epoch_loss = loss(y_pred, y_true)


        epoch_loss.backward()  # Backwards Propagation
        optimizer.step()

    print("-------- Fold", fold, "--------")
    model.eval()
    y_pred = model(torch.tensor(numeric_data.iloc[test_indexes].values, dtype=torch.float32))
    y_pred = torch.reshape(y_pred, (-1,))

    y_true = torch.tensor(data_true.iloc[test_indexes].values, dtype=torch.float32)

    print("Loss:", loss(y_pred, y_true))
    print()

