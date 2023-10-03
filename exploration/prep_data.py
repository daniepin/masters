import os
import re
import numpy as np
import pandas as pd

path = os.path.abspath(os.getcwd())
data_path = os.path.join(path, r"data/ixi/IXI-T1-proc/")
labels_path = os.path.join(path, "data/ixi/labels.npy")
labels_path2 = os.path.join(path, "data/ixi/labels.csv")
files_path = os.path.join(path, "data/ixi/files.npy")
demo_path = os.path.join(path, "exploration/IXI(1).xls")

ixi_demo = pd.read_excel(demo_path, sheet_name=0, header=0, engine="xlrd")

regex = re.compile(r"IXI\d{3}")
rx = re.compile(r"IXI(\d{3})")
files = np.sort(
    [
        os.path.join(data_path, file)
        for file in os.listdir(data_path)
        if not "_mask" in file
    ]
)

imgs = []

files_int_id = np.array([int(rx.search(file).group(1)) for file in files])

ixi_demo = ixi_demo[ixi_demo["IXI_ID"].isin(files_int_id)]
ixi_demo.loc[:, "FILE_ID"] = "IXI" + ixi_demo["IXI_ID"].astype(str).str.zfill(3)


files_id = [regex.search(file).group(0) for file in files]
df = pd.DataFrame()
idx = []

for file in files:
    if (id := regex.search(file).group(0)) in ixi_demo.loc[:, "FILE_ID"].values:
        idx.append(ixi_demo[ixi_demo["FILE_ID"] == id].index.values[0])
        imgs.append(file)

labels = ixi_demo.filter(items=idx, axis=0)

if not os.path.isfile(labels_path2):
    # np.save(labels_path, labels)
    labels.to_csv(labels_path2, index=False)

if not os.path.isfile(files_path):
    np.save(files_path, imgs)
