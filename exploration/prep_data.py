import os
import re
import numpy as np
import pandas as pd
from labels import labels_dict

path = os.path.abspath(os.getcwd())
data_path = os.path.join(path, r"data/ixi/IXI-T1/")
labels_path = os.path.join(path, "data/ixi/labels.npy")
files_path = os.path.join(path, "data/ixi/files.npy")
demo_path = os.path.join(path, "exploration/IXI(1).xls")

ixi_demo = pd.read_excel(demo_path, sheet_name=0, header=0, engine="xlrd")

regex = re.compile(r"IXI\d{3}")
rx = re.compile(r"IXI(\d{3})")
files = np.sort([os.path.join(data_path, file) for file in os.listdir(data_path)])

imgs = []
labels = []

files_int_id = np.array([int(rx.search(file).group(1)) for file in files])

ixi_demo = ixi_demo[ixi_demo["IXI_ID"].isin(files_int_id)]
ixi_demo.loc[:, "FILE_ID"] = "IXI" + ixi_demo["IXI_ID"].astype(str).str.zfill(3)

files_id = [regex.search(file).group(0) for file in files]

for file in files:
    if (id := regex.search(file).group(0)) in ixi_demo.loc[:, "FILE_ID"].values:
        imgs.append(file)
        labels.append(
            ixi_demo[ixi_demo["FILE_ID"] == id]["SEX_ID (1=m, 2=f)"].values[0] - 1
        )

print(labels)
if not os.path.isfile(labels_path):
    np.save(labels_path, labels)

if not os.path.isfile(files_path):
    np.save(files_path, imgs)
