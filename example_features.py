import os
from torch import nn, optim
from torch.utils.data import DataLoader
from catalyst import dl, utils
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

# runner
runner = dl.SupervisedRunner(
    input_key="features", output_key="logits", target_key="targets", loss_key="loss"
)
# training
runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    loaders=loaders,
    num_epochs=1,
    callbacks=[
        dl.AccuracyCallback(input_key="logits", target_key="targets", topk_args=(1, 3, 5)),
        dl.PrecisionRecallF1SupportCallback(
            input_key="logits", target_key="targets", num_classes=10
        ),
        dl.AUCCallback(input_key="logits", target_key="targets"),
    ],
    logdir="./logs",
    valid_loader="valid",
    valid_metric="loss",
    minimize_valid_metric=True,
    verbose=True,
    ddp=False,
    fp16=False,
    load_best_on_end=True,
)
# inference
for prediction in runner.predict_loader(loader=loaders["valid"]):
    assert prediction["logits"].detach().cpu().numpy().shape[-1] == 10

features_batch = next(iter(loaders["valid"]))[0]
# stochastic weight averaging
model.load_state_dict(utils.get_averaged_weights_by_path_mask(logdir="./logs", path_mask="*.pth"))
# tracing
utils.trace_model(model=runner.model, batch=features_batch)
# quantization
utils.quantize_model(model=runner.model)
# pruning
utils.prune_model(model=runner.model, pruning_fn="l1_unstructured", amount=0.8)
# onnx export
utils.onnx_export(model=runner.model, batch=features_batch, file="./logs/mnist.onnx", verbose=True)