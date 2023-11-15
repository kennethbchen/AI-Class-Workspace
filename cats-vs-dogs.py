import torch
import torchvision
from torch import nn
from torchvision import transforms
import torchsummary

img_dimensions = 224
batch_size = 32

# Define the transformations to be applied to the images
img_transforms = transforms.Compose([
    transforms.Resize((img_dimensions, img_dimensions)),
    transforms.ToTensor()
])

# Load the dataset
dataset = torchvision.datasets.ImageFolder(
    root="ignore/cats-vs-dogs",
    transform=img_transforms
)

# Split the dataset into training and validation sets
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [20000, 5000])

# Create training and validation dataloaders
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    num_workers=0,
    shuffle=True,
)

val_dataloader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=batch_size,
    num_workers=0,
    shuffle=True
)

model = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Conv2d(16, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Conv2d(32, 64, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Flatten(),
    nn.Linear(64 * 28 * 28, 512),
    nn.ReLU(),
    nn.Linear(512, 2)
)

#torchsummary.summary(model, (3, img_dimensions, img_dimensions), device="cpu")

optim = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

print("Start Training")
for epoch in range(10):
    for batch in train_dataloader:
        x, y = batch
        y_hat = model(x)
        loss = loss_fn(y_hat, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
    print(f"Epoch {epoch} loss: {loss}")
    correct = 0
    total = 0
    for batch in val_dataloader:
        x, y = batch
        y_hat = model(x)
        _, predicted = torch.max(y_hat.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()
    print(f"Epoch {epoch} accuracy: {correct / total}")