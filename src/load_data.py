import os
import json
from pathlib import Path


datasets = ["ixi", "ukb"]
home = Path.home()

jsons = {
    "ixi": r"datasets/ixi/ixi_dataset.json",
    "ukb": r"datasets/ukb/ukb_dataset.json",
}


def get_data(dataset: str = "ixi", label="sex"):
    if dataset not in datasets:
        print(f"Unfamiliar dataset '{dataset}'")
        exit()

    path = Path(home, jsons[dataset])

    with path.open() as file:
        dataset_json = json.load(file)

    # train_files = [iter["image"] for iter in dataset_json["train"]]
    # val_files = [iter["image"] for iter in dataset_json["val"]]

    # train_labels = [iter[label] for iter in dataset_json["train"]]
    # val_labels = [iter[label] for iter in dataset_json["val"]]

    train = [
        {"image": data["image"], "label": data[label]} for data in dataset_json["train"]
    ]
    val = [
        {"image": data["image"], "label": data[label]} for data in dataset_json["val"]
    ]

    if "test" in dataset_json.keys():
        test = [
            {"image": data["image"], "label": data[label]}
            for data in dataset_json["test"]
        ]
        return {
            "train": train,
            "val": val,
            "test": test,
        }

    return {
        "train": train,
        "val": val,
    }
