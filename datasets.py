import os
import math
import glob
from errno import ENOENT
from enum import Enum, auto

import numpy as np
from PIL import Image
import scipy.ndimage
from torchvision.transforms import ToTensor, Normalize, Compose
from torch.utils.data import Dataset, DataLoader
from augmix import Augmentations, augment_and_mix, normalize, unnormalize

from matplotlib import pyplot as plt


Augmentations.IMAGE_SIZE = 150


class Target(Enum):
    TRAIN = 'train'
    TEST = 'test'
    PRED = 'pred'


labels_to_id = {
    'buildings' : 0,
    'forest'    : 1,
    'glacier'   : 2,
    'mountain'  : 3,
    'sea'       : 4,
    'street'    : 5,
}

J = os.path.join


def read_image(name):
    raw = np.array(Image.open(name))
    if not type(raw) is np.ndarray:
        raise FileNotFoundError(ENOENT, os.strerror(ENOENT), name)
    return raw

class Item():
    def __init__(self, path, label):
        self.path = path
        self.label = label
        self.x = None
        self.y = labels_to_id[label]

    def is_loaded(self):
        return self.x

    def load(self):
        self.x = read_image(self.path)


class BaseDataset(Dataset):
    def __init__(self, mode=Target.TRAIN,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        seg = f'seg_{mode.value}'
        target_dir = J('data', seg, seg)
        self.labels = os.listdir(target_dir)
        self.items = []
        for label in self.labels:
            names = os.listdir(J(target_dir, label))
            for name in sorted(names):
                self.items.append(Item(J(target_dir, label, name), label))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, _idx):
        return self.items[_idx]

if __name__ == '__main__':
    ds = BaseDataset()
    for i, item in enumerate(ds):
        if i != 4:
            continue
        item.load()
        mixed = []
        for _ in range(7):
            a = augment_and_mix(item.x.astype(np.float32))
            # a = unnormalize(a)
            mixed.append(a)
        fig = plt.figure(figsize=(8, 8))
        fig.add_subplot(2, 4, 1)
        plt.imshow(item.x)
        for j, m in enumerate(mixed):
            fig.add_subplot(2, 4, j + 2)
            plt.imshow(m.astype(np.uint8))
        plt.show()
        break
