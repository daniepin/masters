import torch
import monai
import wandb
from load_data import get_data
from transforms import transforms
from model import SFCN
from train import standard_train, vos_train

seed = 2023
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
monai.utils.set_determinism(seed)

"""run = wandb.init(
    project="Medical VOS",
    config={
        "learning_rate": 0.01,
        "architecture": "SFCN",
        "dataset": "IXI",
        "epochs": 10,
    },
)"""


datasets = {
    "ixi": {"path": r"data/ixi/ixi_dataset.json", "label": "sex"},
    "ukb": {"path": r"ukb/ukb_dataset.json", "label": "sex"},
}

use_dataset = "ixi"
vos = False

state = {
    "current_epoch": 0,
    "current_loss": 0.0,
    "best_loss": 0.0,
    "best_accuracy": 0.0,
    "best_epoch": 0,
}

params = {
    "use_gpu": True,
    "batch_size": 5,
    "num_classes": 2,
    "output_channels": 2,
    "epochs": 140,
    "decay": 0.0005,
    "lr": 0.01,
    "momentum": 0.9,
    "optimizer": "SGD",
    "samples": 100,  # 1000
    "start_epoch": 2,  # 40
    "sample_from": 1000,
    "select": 1,
    "loss_weight": 0.1,
    "vos_enable": False,
    "remote": False,
    "ngpus": 1,
}

by_reference = {}  # {"wandb": run}
vos_params = {}


def main() -> None:
    device = torch.device(
        "cuda" if params["use_gpu"] and torch.cuda.is_available() else "cpu"
    )
    params["device"] = device

    data = get_data(
        use_dataset,
        fpath=datasets[use_dataset]["path"],
        label=datasets[use_dataset]["label"],
    )

    print(f"Size of training data: {len(data['train'][0])}")
    print(f"Size of validation data: {len(data['val'][0])}")

    train_dataset = monai.data.ImageDataset(
        data["train"][0][:10],
        labels=data["train"][1][:10],
        transform=transforms[use_dataset]["train"],
    )

    train_loader = monai.data.DataLoader(
        train_dataset,
        batch_size=params["batch_size"],
        num_workers=8,
        pin_memory=torch.cuda.is_available(),
        shuffle=True,
    )

    val_dataset = monai.data.ImageDataset(
        data["val"][0],
        labels=data["val"][1],
        transform=transforms[use_dataset]["val"],
    )

    val_loader = monai.data.DataLoader(
        val_dataset,
        batch_size=params["batch_size"],
        num_workers=8,
        pin_memory=torch.cuda.is_available(),
    )

    model = SFCN(1, [32, 64, 128, 256, 256, 64], 2)
    if params["ngpus"] > 1:
        model = torch.nn.DataParallel(model, device_ids=[range(params["ngpus"])])
    else:
        model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params["lr"],
        weight_decay=params["decay"],
    )

    loss_criterion = torch.nn.CrossEntropyLoss()
    log_reg_criterion = torch.nn.Sequential(
        torch.nn.Linear(1, 12), torch.nn.ReLU(), torch.nn.Linear(12, 2)
    ).to(device)

    by_reference["model"] = model
    by_reference["optimizer"] = optimizer
    by_reference["train_loader"] = train_loader
    by_reference["val_loader"] = val_loader
    by_reference["loss_criterion"] = loss_criterion
    by_reference["log_reg_criterion"] = log_reg_criterion

    print(f"Using params: {params}")
    # print(f"Passing variables as refrence: {by_reference}")
    print(f"Training vos is : {vos}")

    if vos:
        vos_train(by_reference, params, state)
    else:
        standard_train(by_reference, params, state)

    print(state)

    """best = 0
    for epoch in range(params["epochs"]):
        model.train()
        running_loss = 0

        for data, target in train_loader:
            print(target.device)
            data, target = data.to(device), target.to(device)

            # sum_temp = sum(classes_dict.values())

            # if sum_temp == num_classes * samples:
            #   energy_regularization()

            optimizer.zero_grad()

            outputs = model(data)
            loss = loss_criterion(outputs, target)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{params['epochs']}] - Training Loss: {avg_loss:.4f}")

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)

                _, predicted = torch.max(outputs, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        validation_accuracy = 100 * correct / total
        best = max(best, validation_accuracy)
        print(
            f"Epoch [{epoch+1}/{params['epochs']}] - Validation Accuracy: {validation_accuracy:.2f}%"
        )

    print(f"Best achieved: {best}")"""


if __name__ == "__main__":
    main()
