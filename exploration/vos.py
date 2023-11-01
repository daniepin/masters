import os
import torch
import monai
import numpy as np
import pandas as pd
import monai.transforms as mts
from vos_utils import train, test
from sklearn.model_selection import train_test_split
from model import SFCN
from torch.utils.tensorboard import SummaryWriter

seed = 2023
rs = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(seed)))
torch.manual_seed(seed)


def get_file_paths():
    return np.load("data/ixi/files.npy")


def get_labels():
    return pd.read_csv("data/ixi/labels.csv")


def split_data(images, labels, random_state):
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels, train_size=0.5, random_state=random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_val, y_val, train_size=0.5, random_state=random_state
    )

    return {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
    }


def main():
    files = get_file_paths()
    labels = get_labels()["SEX_ID (1=m, 2=f)"].to_numpy() - 1
    num_classes = 2

    data_split = split_data(files, labels, rs)
    print(f"Size of training data: {len(data_split['train'][0])}")
    print(f"Size of validation data: {len(data_split['val'][0])}")
    print(f"Size of test data: {len(data_split['test'][0])}")

    train_transforms = mts.Compose(
        [
            mts.EnsureChannelFirst(),
            mts.CropForeground(),
            mts.Spacing(pixdim=[2, 2, 2], mode="bilinear"),
            mts.ResizeWithPadOrCrop(40, mode="constant", value=0.0),
            mts.ScaleIntensity(minv=0.0, maxv=1.0),
        ]
    )
    val_transforms = mts.Compose(
        [
            mts.EnsureChannelFirst(),
            mts.CropForeground(),
            mts.RandFlip(prob=0.5, spatial_axis=0),
            mts.Spacing(pixdim=[2, 2, 2], mode="bilinear"),
            mts.ResizeWithPadOrCrop(
                spatial_size=(40 + 5, 40 + 5, 40 + 5), mode="constant", value=0.0
            ),
            mts.RandSpatialCrop(roi_size=40, random_center=True, random_size=False),
            mts.ScaleIntensity(minv=0.0, maxv=1.0),
        ]
    )
    train_dataset = monai.data.ImageDataset(
        data_split["train"][0],
        labels=data_split["train"][1],
        transform=train_transforms,
    )

    train_loader = monai.data.DataLoader(
        train_dataset,
        batch_size=10,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

    val_dataset = monai.data.ImageDataset(
        data_split["val"][0], labels=data_split["val"][1], transform=val_transforms
    )
    val_loader = monai.data.DataLoader(
        val_dataset, batch_size=10, num_workers=4, pin_memory=torch.cuda.is_available()
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 0:
        pass

    model = SFCN(1, [32, 64, 128, 256, 256, 64], 2).to(device)

    epochs = 60
    decay = 0.0005
    lr = 0.01
    momentum = 0.9

    weight_energy = torch.nn.Linear(num_classes, 1).to(device)
    torch.nn.init.uniform_(weight_energy.weight)

    logistic_regression = torch.nn.Sequential(
        torch.nn.Linear(1, 12), torch.nn.ReLU(), torch.nn.Linear(12, 2)
    )
    logistic_regression = logistic_regression.to(device)

    optimizer = torch.optim.SGD(
        list(model.parameters())
        + list(weight_energy.parameters())
        + list(logistic_regression.parameters()),
        lr=lr,
        momentum=momentum,
        weight_decay=decay,
        nesterov=True,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, epochs * len(train_loader), 1e-6 / lr, -1
    )

    writer = SummaryWriter()
    best = 0
    best_epoch = 0
    best_state = None
    for epoch in range(0, epochs):
        print(f"Current epoch: {epoch}")
        train(
            model,
            train_loader,
            epoch,
            optimizer,
            scheduler,
            logistic_regression,
            device,
            writer,
            weight_energy,
        )
        accuracy = test(model, val_loader, epoch, device, writer)
        print(f"Current accuracy: {accuracy}")

        if accuracy > best:
            best = accuracy
            best_epoch = epoch
            best_state = model.state_dict().copy()

    torch.save(
        best_state,
        os.path.join(
            r"exploration/result", rf"best_model_{best_epoch}_ac{int(best)}.pt"
        ),
    )
    print(f"Best accuracy achieved: {best}")
    print(f"During epoch: {best_epoch}")


if __name__ == "__main__":
    main()
