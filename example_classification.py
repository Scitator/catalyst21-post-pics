import os
from torch import nn, optim
from torch.utils.data import DataLoader
from catalyst import dl
from catalyst.data.transforms import ToTensor
from catalyst.contrib.datasets import MNIST

# data
loaders = {
    "train": DataLoader(
        MNIST(os.getcwd(), train=True, download=True, transform=ToTensor()), batch_size=32
    ),
    "valid": DataLoader(
        MNIST(os.getcwd(), train=False, download=True, transform=ToTensor()), batch_size=32
    ),
}

# model, criterion, optimizer, scheduler
model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.02)

runner = dl.SupervisedRunner()
# training
runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    loaders=loaders,
    num_epochs=1,
    logdir="./logs",
    valid_loader="valid",
    valid_metric="loss",
    minimize_valid_metric=True,
    verbose=True,
    callbacks=[
        dl.AccuracyCallback(input_key="logits", target_key="targets", num_classes=10),
        dl.PrecisionRecallF1SupportCallback(
            input_key="logits", target_key="targets", num_classes=10
        ),
        dl.AUCCallback(input_key="logits", target_key="targets"),
        dl.ConfusionMatrixCallback(
            input_key="logits", target_key="targets", num_classes=num_classes
        ), 
    ]
)