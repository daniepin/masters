import os
import torch
import monai
import numpy as np
from vos_train import train
from vos_utils import test
from src.transforms import ukb_transforms
from model import SFCN
from main import get_file_paths, get_labels
from utility import split_data

# from test_ukb import train_labels, train_files, val_files, val_labels

seed = 2023
rs = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(seed)))
torch.manual_seed(seed)

state = {
    "epochs": 40,
    "decay": 0.0005,
    "lr": 0.01,
    "momentum": 0.9,
    "num_classes": 2,
    "batch_size": 10,
    "optimizer": "SGD",
    "sample_number": 100,  # 1000
    "start_epoch": 2,  # 40
    "sample_from": 1000,
    "select": 1,
    "loss_weight": 0.1,
    "vos_enable": False,
    "remote": False,
}


def main():
    if state["remote"]:
        try:
            from test_ukb import train_labels, train_files, val_files, val_labels
        except:
            print("Could not import ukb functions")
    else:
        files = get_file_paths()
        labels = get_labels()["SEX_ID (1=m, 2=f)"].to_numpy() - 1

        data_split = split_data(files, labels, rs, test=False)
        train_files, train_labels = data_split["train"]
        val_files, val_labels = data_split["train"]

    print(f"Size of training data: {len(train_files)}")
    print(f"Size of validation data: {len(val_files)}")

    train_transform, val_transform = ukb_transforms()

    train_dataset = monai.data.ImageDataset(
        train_files,
        labels=train_labels,
        transform=train_transform,
    )

    train_loader = monai.data.DataLoader(
        train_dataset,
        batch_size=state["batch_size"],
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

    val_dataset = monai.data.ImageDataset(
        val_files, labels=val_labels, transform=val_transform
    )
    val_loader = monai.data.DataLoader(
        val_dataset,
        batch_size=state["batch_size"],
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 0:
        pass

    model = SFCN(1, [32, 64, 128, 256, 256, 64], 2).to(device)

    optim_params = []
    vos_params = None
    if state["vos_enable"]:
        vos_params = {
            "weight_energy": torch.nn.Linear(state["num_classes"], 1).to(device),
            "log_reg": torch.nn.Sequential(
                torch.nn.Linear(1, 12), torch.nn.ReLU(), torch.nn.Linear(12, 2)
            ),
        }

        torch.nn.init.uniform_(vos_params["weight_energy"].weight)
        vos_params["log_reg"].to(device)

        optim_params = list(vos_params["weight_energy"].parameters()) + list(
            vos_params["log_reg"].parameters()
        )

    optimizer = torch.optim.SGD(
        list(model.parameters()) + optim_params,
        lr=state["lr"],
        weight_decay=state["decay"],
        nesterov=True,
        momentum=0.9,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, state["epochs"] * len(train_loader), 1e-6 / state["lr"], -1
    )

    best = 0
    best_epoch = 0
    best_state = None
    for epoch in range(0, state["epochs"]):
        print(f"Current epoch: {epoch}")
        train(
            model, state, train_loader, epoch, optimizer, scheduler, device, vos_params
        )

        accuracy = test(model, val_loader, epoch, device)
        print(f"Current accuracy: {accuracy}")

        if accuracy > best:
            best = accuracy
            best_epoch = epoch
            # best_state = model.state_dict().copy()

    # torch.save(
    #    best_state,
    #    os.path.join(
    #        r"result", rf"best_model_{best_epoch}_ac{int(best)}.pt"
    #    ),
    # )
    print(f"Best accuracy achieved: {best}")
    print(f"During epoch: {best_epoch}")


if __name__ == "__main__":
    main()
