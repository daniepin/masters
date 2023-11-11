import os
import json

datasets = ["ixi", "ukb"]


def get_data(
    dataset: str = "ixi", fpath: str = r"data/ixi/ixi_dataset.json", label="sex"
):
    if dataset not in datasets:
        print(f"Could not find dataset '{dataset}'")
        exit()

    if dataset == "ukb":
        fpath = r"ukb/ukb_dataset.json"

    with open(fpath) as file:
        dataset_json = json.load(file)

    train_files = [iter["image"] for iter in dataset_json["train"]]
    val_files = [iter["image"] for iter in dataset_json["val"]]

    train_labels = [iter[label] for iter in dataset_json["train"]]
    val_labels = [iter[label] for iter in dataset_json["val"]]

    if "test" in dataset_json.keys():
        test_files = [iter["image"] for iter in dataset_json["test"]]
        test_labels = [iter[label] for iter in dataset_json["test"]]

        return {
            "train": (train_files, train_labels),
            "val": (val_files, val_labels),
            "test": (test_files, test_labels),
        }

    return {
        "train": (train_files, train_labels),
        "val": (val_files, val_labels),
    }
