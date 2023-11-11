import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(images, labels, random_state, test=True):
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels, train_size=0.7, random_state=random_state
    )

    if not test:
        return {
            "train": (X_train, y_train),
            "val": (X_val, y_val),
        }

    X_val, X_test, y_val, y_test = train_test_split(
        X_val, y_val, train_size=0.5, random_state=random_state
    )

    return {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
    }


def get_file_paths():
    return np.load("data/ixi/files.npy")


def get_labels():
    # return np.load("data/ixi/labels.npy")
    return pd.read_csv("data/ixi/labels.csv")
