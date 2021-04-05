import os
import optuna
import torch
from torch import nn
from torch.utils.data import DataLoader
from catalyst import dl
from catalyst.data.transforms import ToTensor
from catalyst.contrib.datasets import MNIST
    

def objective(trial):
    lr = trial.suggest_loguniform("lr", 1e-3, 1e-1)
    num_hidden = int(trial.suggest_loguniform("num_hidden", 32, 128))

    loaders = {
        "train": DataLoader(
            MNIST(os.getcwd(), train=True, download=True, transform=ToTensor()), batch_size=32
        ),
        "valid": DataLoader(
            MNIST(os.getcwd(), train=False, download=True, transform=ToTensor()), batch_size=32
        ),
    }
    model = nn.Sequential(
        nn.Flatten(), nn.Linear(784, num_hidden), nn.ReLU(), nn.Linear(num_hidden, 10)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    runner = dl.SupervisedRunner(input_key="features", output_key="logits", target_key="targets")
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        loaders=loaders,
        callbacks={
            "accuracy": dl.AccuracyCallback(
                input_key="logits", target_key="targets", num_classes=10
            ),
            "optuna": dl.OptunaPruningCallback(
                loader_key="valid", metric_key="accuracy01", minimize=False, trial=trial
            ),
        },
        num_epochs=3,
    )
    score = runner.callbacks["optuna"].best_score
    return score

study = optuna.create_study(
    direction="maximize",
    pruner=optuna.pruners.MedianPruner(
        n_startup_trials=1, n_warmup_steps=0, interval_steps=1
    ),
)
study.optimize(objective, n_trials=3, timeout=300)
print(study.best_value, study.best_params)