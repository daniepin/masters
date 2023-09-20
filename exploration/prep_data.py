import os
import re
import numpy as np
from labels import labels_dict

path = os.path.abspath(os.getcwd())
data_path = os.path.join(path, r"data/ixi/IXI-T1/")
labels_path = os.path.join(path, "data/ixi/labels.npy")
files_path = os.path.join(path, "data/ixi/files.npy")


regex = re.compile(r"IXI\d{3}")
files = np.sort([os.path.join(data_path, file) for file in os.listdir(data_path)])

imgs = []
labels = []

files_id = [regex.search(file).group(0) for file in files]

idx = 0
for key in files_id:
    if key in labels_dict.keys():
        imgs.append(files[idx])
        labels.append(labels_dict[key])
    idx += 1

if not os.path.isfile(labels_path):
    np.save(labels_path, labels)

if not os.path.isfile(files_path):
    np.save(files_path, imgs)
