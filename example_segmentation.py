import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from catalyst import dl
from catalyst.data.transforms import ToTensor
from catalyst.contrib.datasets import MNIST
from catalyst.contrib.nn import IoULoss

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
model = nn.Sequential(
    nn.Conv2d(1, 1, 3, 1, 1), nn.ReLU(), 
    nn.Conv2d(1, 1, 3, 1, 1), nn.Sigmoid(),
)
criterion = IoULoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

# runner
class CustomRunner(dl.SupervisedRunner):
    def handle_batch(self, batch):
        x = batch[self._input_key]
        x_noise = (x + torch.rand_like(x)).clamp_(0, 1)
        x_ = self.model(x_noise)
        self.batch = {self._input_key: x, self._output_key: x_, self._target_key: x}

runner = CustomRunner(
    input_key="features", output_key="scores", target_key="targets", loss_key="loss"
)
# training
runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    loaders=loaders,
    num_epochs=1,
    callbacks=[
        dl.IOUCallback(input_key="scores", target_key="targets"),
        dl.DiceCallback(input_key="scores", target_key="targets"),
        dl.TrevskyCallback(input_key="scores", target_key="targets", alpha=0.2),
    ],
    logdir="./logdir",
    valid_loader="valid",
    valid_metric="loss",
    minimize_valid_metric=True,
    verbose=True,
)