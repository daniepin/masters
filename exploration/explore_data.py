import matplotlib.pyplot as plt
import torch
import torchvision
import monai
import torchio
import os
import numpy as np
import nibabel as nib
from model import SFCN
from torch.utils.tensorboard import SummaryWriter

import monai.transforms as mts

# monai.config.print_config()

path = os.path.abspath(os.getcwd())
data_path = os.path.join(path, r"data/ixi/IXI-T1/")


images = [
    "IXI314-IOP-0889-T1.nii.gz",
    "IXI249-Guys-1072-T1.nii.gz",
    "IXI609-HH-2600-T1.nii.gz",
    "IXI173-HH-1590-T1.nii.gz",
    "IXI020-Guys-0700-T1.nii.gz",
    "IXI342-Guys-0909-T1.nii.gz",
    "IXI134-Guys-0780-T1.nii.gz",
    "IXI577-HH-2661-T1.nii.gz",
    "IXI066-Guys-0731-T1.nii.gz",
    "IXI130-HH-1528-T1.nii.gz",
    "IXI607-Guys-1097-T1.nii.gz",
    "IXI175-HH-1570-T1.nii.gz",
    "IXI385-HH-2078-T1.nii.gz",
    "IXI344-Guys-0905-T1.nii.gz",
    "IXI409-Guys-0960-T1.nii.gz",
    "IXI584-Guys-1129-T1.nii.gz",
    "IXI253-HH-1694-T1.nii.gz",
    "IXI092-HH-1436-T1.nii.gz",
    "IXI574-IOP-1156-T1.nii.gz",
    "IXI585-Guys-1130-T1.nii.gz",
]

labels = torch.tensor([0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0])

# labels = torch.nn.functional.one_hot(labels, 2)

# files = [os.path.join(data_path, file) for file in os.listdir(data_path)]
files = [os.path.join(data_path, file) for file in images]

transforms = mts.Compose(
    [mts.ScaleIntensity(), mts.EnsureChannelFirst(), mts.Resize((40, 40, 40))]
)
dataset = monai.data.ImageDataset(files[0:6], labels=labels[0:6], transform=transforms)


# Perform basic checks
loader: torch.utils.data.DataLoader = monai.data.DataLoader(
    dataset, batch_size=2, num_workers=4, pin_memory=torch.cuda.is_available()
)
im, label = monai.utils.misc.first(loader)
print(type(im), im.shape, label)


train_loader = monai.data.DataLoader(
    dataset, batch_size=3, num_workers=4, pin_memory=torch.cuda.is_available()
)

val_dataset = monai.data.ImageDataset(
    files[6:12], labels=labels[6:12], transform=transforms
)
val_loader = monai.data.DataLoader(
    val_dataset, batch_size=2, num_workers=4, pin_memory=torch.cuda.is_available()
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SFCN(1, [32, 64, 128, 256, 256, 64], 2).to(device)
# model = monai.networks.nets.DenseNet121(
#    spatial_dims=3, in_channels=1, out_channels=2).to(device)
# print(model)
loss_function = torch.nn.CrossEntropyLoss()
# loss_function = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), 1e-5)


val_interval = 2
best_metric = -1
epoch_loss_values = list()
metric_values = list()
writer = SummaryWriter()
for epoch in range(15):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{5}")
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in train_loader:
        step += 1
        inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_len = len(dataset) // train_loader.batch_size
        print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
        writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            num_correct = 0.0
            metric_count = 0
            for val_data in val_loader:
                val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
                val_outputs = model(val_images)
                value = torch.eq(val_outputs.argmax(dim=1), val_labels)
                metric_count += len(value)
                num_correct += value.sum().item()
            metric = num_correct / metric_count
            metric_values.append(metric)
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(
                    model.state_dict(), "best_metric_model_classification3d_array.pth"
                )
                print("saved new best metric model")
            print(
                "current epoch: {} current accuracy: {:.4f} best accuracy: {:.4f} at epoch {}".format(
                    epoch + 1, metric, best_metric, best_metric_epoch
                )
            )
            writer.add_scalar("val_accuracy", metric, epoch + 1)
print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
writer.close()


"""
img = nib.load(files[0])
img_data = img.get_fdata()
for i in range(img_data.shape[2]):
    plt.figure()
    plt.imshow((img_data[:, :, i]), cmap="gray")
    plt.title(f"slice {i}")
    plt.axis('off')  # Turn off axis labels
    # Pause to display each slice (adjust the pause duration as needed)
    plt.pause(0.1)

    output_filename = os.path.join(
        "/home/neutron/dev/thesis/exploration/brain", f'slice_{i}.png')
    plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)
    plt.close()
"""
