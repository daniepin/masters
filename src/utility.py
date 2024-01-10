import os
import pathlib
import monai
import torch
import wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transforms import get_transforms
from sklearn.model_selection import train_test_split

home = pathlib.Path.home().as_posix()

def split_data(images, labels, random_state, test=True):
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels, train_size=0.7, random_state=random_state
    )

    if not test:
        return {
            "train": (X_train, y_train),
            "val": (X_val, y_val),
        }

    X_val, X_test, y_val, y_test = train_test_split(
        X_val, y_val, train_size=0.5, random_state=random_state
    )

    return {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
    }


def get_file_paths(path: str):
    return np.load(path)


def get_labels(path: str):
    return pd.read_csv(path)


def create_loaders(data, use_dataset, params):
    transforms = get_transforms(use_dataset, params["image_size"], params["pixdim"])
    
    folder = r""
    if use_dataset == "ukb":
        folder = r"/mnt/scratch/daniel/datasets/ukb_preprocessed/bids/"

    data["train"] = [
        {"image": os.path.join(home, folder + i["image"]), "label": int(i["label"])}
        for i in data["train"]
    ]

    data["val"] = [
        {"image": os.path.join(home, folder + i["image"]), "label": int(i["label"])}
        for i in data["val"]
    ]

    data["test"] = [
        {"image": os.path.join(home, folder + i["image"]), "label": int(i["label"])}
        for i in data["test"]
    ]

    train_dataset = monai.data.Dataset(
        data=data["train"][:30],
        transform=transforms["train"],
    )

    train_loader = monai.data.DataLoader(
        train_dataset,
        batch_size=params["batch_size"],
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        shuffle=True,
    )

    val_dataset = monai.data.Dataset(
        data=data["val"][:10],
        transform=transforms["val"],
    )

    val_loader = monai.data.DataLoader(
        val_dataset,
        batch_size=params["batch_size"],
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

    test_dataset = monai.data.Dataset(
        data=data["test"][:5],
        transform=transforms["val"],
    )

    test_loader = monai.data.DataLoader(
        test_dataset,
        batch_size=params["batch_size"],
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader


def view_image(loader, fname: str, device):
    data_first = monai.utils.first(loader)
    data_first["image"].to(device)
    print(
        f"image shape: {data_first['image'].shape}, label shape: {data_first['label'].shape}"
    )
    _, _, x, y, z = data_first["image"].shape
    x_ = x // 2
    y_ = y // 2
    z_ = z // 2
    img1 = data_first["image"][0, 0, x_, :, :]
    img2 = data_first["image"][0, 0, :, y_, :]
    img3 = data_first["image"][0, 0, :, :, z_]
    comb = torch.cat((img2, img3), 1)

    black = torch.zeros(img1.shape[0], comb.shape[1] - img1.shape[1])
    comb2 = torch.cat((img1, black), 1)
    combined = torch.cat((comb, comb2), 0)
    # Set all negative values to zero
    torch.nn.functional.relu(combined, inplace=True)

    plt.imshow(combined, cmap="gray")
    plt.axis("off")  # Turn off axis labels
    # plt.suptitle("Brain MRI overview", y=0.745)
    plt.savefig(
        os.path.join(home, fname),
        bbox_inches="tight",
        pad_inches=0.0,
    )
    wandb.log({f"{fname}": wandb.Image(os.path.join(home, fname))})


def save_model(path, model, optimizer, epoch, loss, name, acc=None, upload=None):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "acc": acc
        },
        path,
    )

    if upload:
        artifact = wandb.Artifact(name, type='model')
        artifact.add_file(path)
        wandb.log_artifact(artifact)
