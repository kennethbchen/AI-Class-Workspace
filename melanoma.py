import torch
import torchvision
from torch import nn
from torchvision import transforms
import torchsummary

def main():
    # Checks if GPU can be used
    # https://pytorch.org/get-started/locally/
    if torch.cuda.is_available():
        print("Cuda is available")
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    img_dimensions = 224
    batch_size = 512

    # Define the transformations to be applied to the images
    img_transforms = transforms.Compose([
        transforms.Resize((img_dimensions, img_dimensions)),
        transforms.ToTensor()
    ])

    print("Load dataset")

    """
    # Load the dataset
    dataset = torchvision.datasets.ImageFolder(
        root="ignore/cats-vs-dogs",
        transform=img_transforms
    )
    """

    train_folder_data = torchvision.datasets.ImageFolder(
        root="ignore/melanoma-benign-vs-malignant/train",
        transform=img_transforms
    )

    test_dataset = torchvision.datasets.ImageFolder(
        root="ignore/melanoma-benign-vs-malignant/test",
        transform=img_transforms
    )


    # Split the train dataset into training and validation sets
    train_dataset, val_dataset = torch.utils.data.random_split(train_folder_data, [9000, 605])



    # Create training and validation dataloaders
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=2,
        shuffle=True,
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=2,
        shuffle=True
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=2,
        shuffle=True
    )

    model = torchvision.models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(2048, 2)
    model.to(device)

    #torchsummary.summary(model, (3, img_dimensions, img_dimensions), device="cpu")

    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    print("Start Training")
    for epoch in range(10):

        for batch in train_dataloader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)

            y_hat = model(x)
            y_hat.to(device)

            loss = loss_fn(y_hat, y)

            optim.zero_grad()
            loss.backward()
            optim.step()
            print("Batch Complete")
        print(f"Epoch {epoch} loss: {loss}")
        print("Testing Epoch accuracy")
        correct = 0
        total = 0
        for batch in val_dataloader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            y_hat = model(x)
            y_hat.to(device)
            _, predicted = torch.max(y_hat.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
        print(f"Epoch {epoch} accuracy: {correct / total}")


if __name__ == '__main__':
    main()