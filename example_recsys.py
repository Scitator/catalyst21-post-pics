import torch
from torch.utils.data import DataLoader, TensorDataset
from catalyst import dl

# data
num_users, num_features, num_items = int(1e4), int(1e1), 10
X = torch.rand(num_users, num_features)
y = (torch.rand(num_users, num_items) > 0.5).to(torch.float32)
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32, num_workers=1)
loaders = {"train": loader, "valid": loader}

# model, criterion, optimizer, scheduler
model = torch.nn.Linear(num_features, num_items)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters())
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [2])

# runner
class CustomRunner(dl.Runner):
    def handle_batch(self, batch):
        x, y = batch
        logits = self.model(x)
        self.batch = {
            "features": x, "logits": logits, "scores": torch.sigmoid(logits), "targets": y
        }

# training
runner = CustomRunner()
runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    loaders=loaders,
    num_epochs=3,
    verbose=True,
    callbacks=[
        dl.CriterionCallback(input_key="logits", target_key="targets", metric_key="loss"),
        dl.AUCCallback(input_key="scores", target_key="targets"),
        dl.HitrateCallback(input_key="scores", target_key="targets", topk_args=(1, 3, 5)),
        dl.MRRCallback(input_key="scores", target_key="targets", topk_args=(1, 3, 5)),
        dl.MAPCallback(input_key="scores", target_key="targets", topk_args=(1, 3, 5)),
        dl.NDCGCallback(input_key="scores", target_key="targets", topk_args=(1, 3, 5)),
        dl.OptimizerCallback(metric_key="loss"),
        dl.SchedulerCallback(),
        dl.CheckpointCallback(
            logdir="./logs", loader_key="valid", metric_key="map01", minimize=False
        ),
    ]
)