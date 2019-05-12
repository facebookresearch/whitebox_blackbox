from __future__ import print_function
from PIL import Image
import os
import os.path
from os.path import join
import numpy as np
import sys
import torchvision

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

def createSplits(root, mode, sz_splits, sz_blocks):
    root_mode = join(root, "cifar-dejavu", mode)
    os.makedirs(root_mode)

    base_folder = 'cifar-10-batches-py'
    train_files = ['data_batch_1', 'data_batch_2', 'data_batch_3',
        'data_batch_4', 'data_batch_5']
    all_data = []
    all_labels = []
    for f in train_files:
        fname = os.path.join(root, base_folder, f)
        fo = open(fname, 'rb')
        if sys.version_info[0] == 2:
            entry = pickle.load(fo)
        else:
            entry = pickle.load(fo, encoding='latin1')
        all_data.append(entry['data'])
        if 'labels' in entry:
            all_labels += entry['labels']
        else:
            all_labels += entry['fine_labels']
        fo.close()

    all_data = np.concatenate(all_data)
    all_data = all_data.reshape((50000, 3, 32, 32))
    all_data = all_data.transpose((0, 2, 3, 1))  # convert to HWC
    all_labels = np.array(all_labels)

    idx = np.argsort(all_labels)
    all_data = all_data[idx]
    all_labels = all_labels[idx]

    # Arange labels like this: [0, 1, ..., 9, 0, 1, ..., 8, 9]
    nlabels = 1 + np.max(all_labels)
    idx = np.arange(all_labels.shape[0]).reshape(nlabels, -1).T.flatten()
    all_data = all_data[idx]
    all_labels = all_labels[idx]

    data, labels = {}, {}

    seen = 0
    for k_split in sz_splits.keys():
        for k_block in sz_blocks.keys():
            key = k_split + "-" + k_block
            sz = sz_splits[k_split] * sz_blocks[k_block] * all_data.shape[0]
            print(key, sz)
            sz = int(sz)
            data[key] = all_data[seen:(seen+sz)]
            labels[key] = all_labels[seen:(seen+sz)]

            seen += sz

    for k in data.keys():
        np.save(join(root_mode, "data-%s" % k), data[k])
        np.save(join(root_mode, "label-%s" % k), labels[k])
        print(k, data[k].shape, labels[k].shape)

    for k_split in sz_splits.keys():
        keys = [k_split + "-" + k_block for k_block in sz_blocks.keys()]
        k_data = np.concatenate([data[k] for k in keys])
        k_label = np.concatenate([labels[k] for k in keys])

        print(k_split, k_data.shape, k_label.shape)

        np.save(join(root_mode, "data-%s" % k_split), k_data)
        np.save(join(root_mode, "label-%s" % k_split), k_label)

    for k_block in sz_blocks.keys():
        keys = [k_split + "-" + k_block for k_split in sz_splits.keys()]
        k_data = np.concatenate([data[k] for k in keys])
        k_label = np.concatenate([labels[k] for k in keys])

        print(k_block, k_data.shape, k_label.shape)
        np.save(join(root_mode, "data-%s" % k_block), k_data)
        np.save(join(root_mode, "label-%s" % k_block), k_label)


def createSplitsCifar100(root, mode, sz_splits, sz_blocks):
    root_mode = join(root, "cifar100-dejavu", mode)
    print("Storing splits in %s" % root_mode)
    os.makedirs(root_mode)

    base_folder = 'cifar-100-python'
    train_files = ['train']

    all_data = []
    all_labels = []
    for f in train_files:
        fname = os.path.join(root, base_folder, f)
        fo = open(fname, 'rb')
        if sys.version_info[0] == 2:
            entry = pickle.load(fo)
        else:
            entry = pickle.load(fo, encoding='latin1')
        all_data.append(entry['data'])
        if 'labels' in entry:
            all_labels += entry['labels']
        else:
            all_labels += entry['fine_labels']
        fo.close()

    all_data = np.concatenate(all_data)
    all_data = all_data.reshape((50000, 3, 32, 32))
    all_data = all_data.transpose((0, 2, 3, 1))  # convert to HWC
    all_labels = np.array(all_labels)

    idx = np.argsort(all_labels)
    all_data = all_data[idx]
    all_labels = all_labels[idx]

    # Arange labels like this: [0, 1, ..., 9, 0, 1, ..., 8, 9]
    nlabels = 1 + np.max(all_labels)
    idx = np.arange(all_labels.shape[0]).reshape(nlabels, -1).T.flatten()
    all_data = all_data[idx]
    all_labels = all_labels[idx]

    data, labels = {}, {}

    seen = 0
    for k_split in sz_splits.keys():
        for k_block in sz_blocks.keys():
            key = k_split + "-" + k_block
            sz = sz_splits[k_split] * sz_blocks[k_block] * all_data.shape[0]
            print(key, sz)
            sz = int(sz)
            data[key] = all_data[seen:(seen+sz)]
            labels[key] = all_labels[seen:(seen+sz)]

            seen += sz

    for k in data.keys():
        np.save(join(root_mode, "data-%s" % k), data[k])
        np.save(join(root_mode, "label-%s" % k), labels[k])
        print(k, data[k].shape, labels[k].shape)

    for k_split in sz_splits.keys():
        keys = [k_split + "-" + k_block for k_block in sz_blocks.keys()]
        k_data = np.concatenate([data[k] for k in keys])
        k_label = np.concatenate([labels[k] for k in keys])

        print(k_split, k_data.shape, k_label.shape)

        np.save(join(root_mode, "data-%s" % k_split), k_data)
        np.save(join(root_mode, "label-%s" % k_split), k_label)

    for k_block in sz_blocks.keys():
        keys = [k_split + "-" + k_block for k_split in sz_splits.keys()]
        k_data = np.concatenate([data[k] for k in keys])
        k_label = np.concatenate([labels[k] for k in keys])

        print(k_block, k_data.shape, k_label.shape)
        np.save(join(root_mode, "data-%s" % k_block), k_data)
        np.save(join(root_mode, "label-%s" % k_block), k_label)






class CIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, root, mode, split, block=None,
                 transform=None, target_transform=None, select=None,
                 download=False):

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        root_mode = join(self.root, self.dirname(), mode)
        k = split
        if block is not None:
            k += "-" + block

        self.data = np.load(join(root_mode, "data-%s.npy" % k))
        self.labels = np.load(join(root_mode, "label-%s.npy" % k))

        if select is not None:
            print("Selecting %d elements out of %d" % (select.shape[0], self.data.shape[0]))
            # Select is a list of indices
            self.data = self.data[select]
            self.labels = self.labels[select]

        print("Loading CIFAR, mode=%s, split=%s, block=%s, nimg=%d" % (mode, split, block, len(self.data)))

    @classmethod
    def dirname(cls):
        return "cifar-dejavu"


    @classmethod
    def getsize(cls, root, mode, split, block=None):
        root_mode = join(os.path.expanduser(root), cls.dirname(), mode)
        k = split
        if block is not None:
            k += "-" + block

        data = np.load(join(root_mode, "data-%s.npy" % k))

        return data.shape[0]

    @classmethod
    def numclasses(cls):
        return 10

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

        return img, target

    def __len__(self):
        return len(self.data)


class CIFAR100(CIFAR10):
    @classmethod
    def dirname(cls):
        return "cifar100-dejavu"

    @classmethod
    def numclasses(cls):
        return 100
