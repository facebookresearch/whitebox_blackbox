from __future__ import print_function
from PIL import Image
import os
import os.path
from os.path import join
import numpy as np
import torchvision


class CIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, root, name,
                 transform=None, target_transform=None,
                 download=False, spit_index=False):

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        self.data = np.load(join(self.root, "data_%s.npy" % name))
        self.labels = np.load(join(self.root, "label_%s.npy" % name))
        self.spit_index = spit_index

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.spit_index:
            return index, img, target
        else:
            return img, target

    def __len__(self):
        return len(self.data)
