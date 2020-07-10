"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import torch.utils.data as data
import os.path
import numpy as np
from torch.utils.data import DataLoader

def numpy_loader(path):
    data = np.load(path)
    data = data.astype("float32")
    if len(data.shape) > 2:
        data = np.swapaxes(data, 0, 1)
        data = np.swapaxes(data, 1, 2)
    return data

###############################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
###############################################################################

import torch.utils.data as data

from PIL import Image
import os
import os.path

VALID_EXTENSIONS = [
#    '.jpg', '.JPG', '.jpeg', '.JPEG',
#    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.npy',
]


def is_valid_file(filename):
    return any(filename.endswith(extension) for extension in VALID_EXTENSIONS)


def make_dataset(dir):
    dataset = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_valid_file(fname):
                path = os.path.join(root, fname)
                dataset.append(path)

    return dataset


class ImageFolder(data.Dataset):
    def __init__(self, root, transform=None, loader=numpy_loader):
        npys = sorted(make_dataset(root))
        self.root = root
        self.npys = npys
        self.transform = transform
        self.loader = loader
        self.label = {"guitar": 0, "piano": 1, "string": 2}

    def __getitem__(self, index):
        path = self.npys[index]
        npy = self.loader(path)
        npy = self.transform(npy)
        label = self.label[path.split("/")[-2]]
        return npy[0:1], label

    def __len__(self):
        return len(self.npys)
