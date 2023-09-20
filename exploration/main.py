import torch
import monai
import numpy as np
import monai.transforms as mts
from sklearn.model_selection import train_test_split
from model import SFCN
from train import train

seed = 2023
rs = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(seed)))
torch.manual_seed(seed)


def get_file_paths():
    return np.load("data/ixi/files.npy")


def get_labels():
    return np.load("data/ixi/labels.npy")


def split_data(images, labels, random_state):
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels, train_size=0.5, random_state=random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_val, y_val, train_size=0.5, random_state=random_state
    )

    return {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
    }


def main():
    files = get_file_paths()
    labels = get_labels()

    data_split = split_data(files, labels, rs)
    print(f"Size of training data: {len(data_split['train'][0])}")
    print(f"Size of validation data: {len(data_split['val'][0])}")
    print(f"Size of test data: {len(data_split['test'][0])}")

    train_transforms = mts.Compose(
        [mts.ScaleIntensity(), mts.EnsureChannelFirst(), mts.Resize((40, 40, 40))]
    )
    train_dataset = monai.data.ImageDataset(
        data_split["train"][0],
        labels=data_split["train"][1],
        transform=train_transforms,
    )

    train_loader = monai.data.DataLoader(
        train_dataset, batch_size=5, num_workers=4, pin_memory=torch.cuda.is_available()
    )

    val_dataset = monai.data.ImageDataset(
        data_split["val"][0], labels=data_split["val"][1], transform=train_transforms
    )
    val_loader = monai.data.DataLoader(
        val_dataset, batch_size=5, num_workers=4, pin_memory=torch.cuda.is_available()
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SFCN(1, [32, 64, 128, 256, 256, 64], 2).to(device)

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), 1e-5)

    train(
        model, train_loader, val_loader, optimizer, loss_function, train_dataset, device
    )


if __name__ == "__main__":
    main()
