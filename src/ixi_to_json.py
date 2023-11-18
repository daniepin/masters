import os
import json
import numpy as np
from pathlib import Path
from utility import get_file_paths, get_labels, split_data

seed = 2023
rs = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(seed)))


home = Path.home()
ixi_path = os.path.join(home.as_posix(), r"datasets/ixi/")

files = [file for file in get_file_paths(os.path.join(ixi_path, r"files.npy"))]
labels = get_labels(os.path.join(ixi_path, r"labels.csv"))

labels = labels[["IXI_ID", "SEX_ID (1=m, 2=f)", "AGE"]]
labels["SEX_ID (1=m, 2=f)"] = labels["SEX_ID (1=m, 2=f)"] - 1

data = split_data(files, labels, rs)


ixi_dict = {
    "Description": "IXI Dataset",
    "age_labels": {"mean": np.mean(labels["AGE"]), "std": np.std(labels["AGE"])},
    "sex_labels": {"male": 0, "female": 1},
    "train": [],
    "val": [],
    "test": [],
}

for i, file in enumerate(data["train"][0]):
    label = data["train"][1].iloc[i]

    subject = {
        "subject": str(int(label.IXI_ID)).zfill(3),
        "image": file,
        "sex": int(label["SEX_ID (1=m, 2=f)"]),
        "age": label.AGE,
    }

    if np.isnan(subject["age"]):
        print(subject)
        continue

    ixi_dict["train"].append(subject)

for i, file in enumerate(data["val"][0]):
    label = data["val"][1].iloc[i]
    subject = {
        "subject": str(int(label.IXI_ID)).zfill(3),
        "image": file,
        "sex": int(label["SEX_ID (1=m, 2=f)"]),
        "age": label.AGE,
    }

    if np.isnan(subject["age"]):
        print(subject)
        continue

    ixi_dict["val"].append(subject)

for i, file in enumerate(data["test"][0]):
    label = data["test"][1].iloc[i]
    subject = {
        "subject": str(int(label.IXI_ID)).zfill(3),
        "image": file,
        "sex": int(label["SEX_ID (1=m, 2=f)"]),
        "age": float(label.AGE),
    }

    if np.isnan(subject["age"]):
        print(subject)
        continue

    ixi_dict["test"].append(subject)

# print(ixi_dict)
js = json.dumps(ixi_dict)

with open(os.path.join(ixi_path, "ixi_dataset.json"), "w") as out:
    out.write(js)
