import os
import pathlib
import random
import monai
import torch
import wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transforms import get_transforms
from sklearn.metrics import roc_curve, auc
import umap

home = pathlib.Path.home().as_posix()


def get_file_paths(path: str):
    return np.load(path)


def get_labels(path: str):
    return pd.read_csv(path)


def create_loaders(data, use_dataset, params, images=(500, 100, 200), transform=None):
    if transform:
        transforms = transform
    else:
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
        data=data["train"][:images[0]],
        transform=transforms["train"],
    )

    train_loader = monai.data.DataLoader(
        train_dataset,
        batch_size=params["batch_size"],
        num_workers=0,
        pin_memory=False,
        shuffle=True,
        drop_last=True,
    )

    val_dataset = monai.data.Dataset(
        data=data["val"][:images[1]],
        transform=transforms["val"],
    )

    val_loader = monai.data.DataLoader(
        val_dataset,
        batch_size=params["batch_size"],
        num_workers=0,
        pin_memory=False,
    )

    test_dataset = monai.data.Dataset(
        data=data["test"][:images[2]],
        transform=transforms["val"],
    )

    test_loader = monai.data.DataLoader(
        test_dataset,
        batch_size=params["batch_size"],
        num_workers=0,
        pin_memory=False,
    )

    return train_loader, val_loader, test_loader


def view_images(loader, fname: str, device, num_images=4):
    # Convert the loader to a list and flatten it to get all images
    data = [batch for batch in loader]
    # Select num_images random images
    random_images = random.sample(data, num_images)

    combined_images = []
    for i, data_first in enumerate(random_images, 1):
        data_first["image"].to(device)
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
        combined_images.append(combined)

    # Combine all images into a 2xC configuration
    final_image = torch.cat(combined_images, 1)

    plt.imshow(final_image, cmap="gray")
    plt.axis("off")  # Turn off axis labels
    plt.savefig(
        os.path.join(home, f"{fname}.png"),
        bbox_inches="tight",
        pad_inches=0.0,
    )
    wandb.log({f"{fname}": wandb.Image(os.path.join(home, f"{fname}.png"))})


def plot_roc_curve(ground_truth, predicted):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(ground_truth, predicted)
    roc_auc = auc(fpr, tpr)

    # Create a figure
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characterthesisistic')
    plt.legend(loc="lower right")

    path = os.path.join(home, "roc_curve.png")
    plt.savefig(path)

    wandb.log({"roc_curve": wandb.Image(path)})

def plot_roc_curve_local(ground_truth, predicted, prefix=""):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(ground_truth, predicted)
    roc_auc = auc(fpr, tpr)

    # Create a figure
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characterthesisistic')
    plt.legend(loc="lower right")

    path = os.path.join(home, f"{prefix}test_roc_curve.png")
    plt.savefig(path)



def save_model(path, model, optimizer, epoch, loss, name, acc=None, upload=None):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            #"optimizer_vos_state": optimizer[1].state_dict(),
            "loss": loss,
            "acc": acc
        },
        path,
    )

    if upload:
        artifact = wandb.Artifact(name, type='model')
        artifact.add_file(path)
        wandb.log_artifact(artifact)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

class LRPolicy(object):
    def __init__(self, warmup_steps=30):
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        return 1/(self.warmup_steps)*step


def create_umap_plot(tensor, labels):
    data = tensor.detach().cpu().numpy()
    #labels = labels.cpu().numpy()

    reducer = umap.UMAP()
    embedding = reducer.fit_transform(data)

    fig = plt.figure()

    plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='Spectral')
    plt.gca().set_aspect('equal', 'datalim')
    plt.title('UMAP projection', fontsize=24)
    plt.colorbar(label='class')

    fig_name = 'umap.png'
    fig.savefig(fig_name)

    wandb.log({"UMAP": wandb.Image(fig_name)})

    plt.close(fig)

def create_umap_plot_local(tensor, labels):
    data = tensor.detach().cpu().numpy()
    #labels = labels.cpu().numpy()

    reducer = umap.UMAP()
    embedding = reducer.fit_transform(data)

    fig = plt.figure()

    plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='Spectral')
    plt.gca().set_aspect('equal', 'datalim')
    plt.title('UMAP projection', fontsize=24)
    plt.colorbar(label='class')

    fig_name = 'test_umap.png'
    fig.savefig(fig_name)

    #wandb.log({"UMAP": wandb.Image(fig_name)})

    plt.close(fig)