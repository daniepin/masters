import os
import json
import torch
import wandb
import monai
from pathlib import Path
from model import VOS_SFCN
from transforms import get_transforms_ood
from utility import plot_roc_curve_local, create_umap_plot_local
from tqdm import tqdm
import matplotlib.pyplot as plt

home = Path.home().as_posix()
folder = r"/mnt/scratch/daniel/datasets/ukb_preprocessed/bids/"
label = "sex"


def test_model():
    api = wandb.Api()
    run = api.run(path="vosmed/Medical VOS/w1wdudro")

    artifact = api.artifact(r"vosmed/Medical VOS/best_acc:v292", type="model")
    artifact_dir = artifact.download()

    config = run.config
    config["kfold_file"] = "1_fold_split.json"
    model = VOS_SFCN(
        1,
        config["features"],
        2,
        config["num_classes"],
        config["samples"],
        config["beta"],
        config["device"],
    )
    model.to(config["device"])

    state_dict = torch.load(artifact_dir + "/best_acc.pth")
    model.load_state_dict(state_dict["model_state_dict"])

    # Open the JSON file
    dataset_json = r"/home/daniel/thesis/" + f"{config['kfold_file']}"
    with open(dataset_json, "r") as f:
        data = json.load(f)

    # Get the data from the fold_i key
    fold = int(config["fold"]) - 1
    data = data.get(f"fold_{fold}")

    val = [
        {"image": os.path.join(home, folder + i["image"]), "label": 0}
        for i in data["val"]
    ]

    test = [
        {"image": os.path.join(home, folder + i["image"]), "label": 1}
        for i in data["test"]
    ]

    # Create monai.data.Datasets
    val_dataset = monai.data.Dataset(
        data=val[:100],
        transform=get_transforms_ood(config["image_size"], config["pixdim"])["val"],
    )
    test_dataset = monai.data.Dataset(
        data=test[:100],
        transform=get_transforms_ood(config["image_size"], config["pixdim"])["test"],
    )

    dataset = torch.utils.data.ConcatDataset([val_dataset, test_dataset])
    test_loader = monai.data.DataLoader(
        dataset,
        batch_size=config["batch_size"],
        num_workers=0,
        pin_memory=False,
        shuffle=True,
    )

    model.eval()
    # Test the model on the validation and test data:
    correct, total = 0, 0
    roc_predicted = []
    vos_roc_predicted = []
    ground_truth = []
    penultimate_layers = []
    
    with torch.no_grad():
        for idx, data in enumerate(tqdm(test_loader)):
            inputs = data["image"].to(config["device"])
            targets = data["label"].to(config["device"])

            outputs, penultimate_layer = model.forward_virtual(inputs)
            energy = -torch.logsumexp(outputs, dim=1).data.cpu().numpy()
            vos_roc_predicted.extend(energy.tolist())

            if idx % 5 == 0:

                fig, axs = plt.subplots(2, 2, figsize=(10, 10))
                for i, ax in enumerate(axs.flat):
                    slice_idx = inputs[i].shape[-1] // 2
                    ax.imshow(inputs[i].cpu().squeeze().numpy()[:, :, slice_idx], cmap='gray')
                    ax.set_title(f'Energy: {energy[i]:.2f}')
                    ax.axis('off')

                    fig.savefig(f'test_energy_{idx}.png')
                    plt.close(fig)


            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            roc_predicted.extend(torch.softmax(outputs[:, 1], dim=0).tolist())
            ground_truth.extend(targets.cpu())
            penultimate_layers.append(penultimate_layer)

    # Print accuracy
    print(f"Accuracy: {100.0 * correct / total}")
    plot_roc_curve_local(ground_truth, roc_predicted)
    penultimate_layers = torch.cat(penultimate_layers, 0)
    create_umap_plot_local(penultimate_layers, ground_truth)

    return model

test_model()