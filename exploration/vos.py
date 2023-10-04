import numpy as np


def get_labels():
    return np.load("data/ixi/labels.npy")


labels = get_labels()
num_classes = len(labels)
