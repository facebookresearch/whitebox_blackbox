from logging import getLogger
import os
import torch
from torchvision import transforms
from torchvision.datasets import CIFAR100, ImageFolder
from torch.utils.data.distributed import DistributedSampler

from .datasets.cifar import CIFAR10

# from .datasets import TripletDataset, LimitDataset
# from .datasets.folder import ImageFolder, EmbeddingsFolder
# from .datasets.sampler import BalancedSampler, DistributedSampler, OrderedSampler
# from .imagenet22k import Imagenet22kDataset, IMAGENET22K_DIR


DATASETS = {
    'imagenet': {
        'train': '/datasets01_101/imagenet_full_size/061417/train',
        'valid': '/datasets01_101/imagenet_full_size/061417/val',
        'num_classes': 1000,
    },
    'cifar10': '/private/home/asablayrolles/data/cifar-dejalight2/'
}

logger = getLogger()
NORMALIZE_IMAGENET = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
NORMALIZE_CIFAR = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

def getCifarTransform(name, img_size=40, crop_size=32, as_list=False, normalization=True):
    assert name in ["center", "random"]

    if name == "random":
        transform = [
            transforms.RandomCrop(crop_size, padding=(img_size - crop_size) // 2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]
    else:
        assert crop_size == 32
        transform = [
            transforms.ToTensor()
        ]

    if normalization:
        transform.append(NORMALIZE_CIFAR)

    if as_list:
        return transform
    else:
        return transforms.Compose(transform)


def getImagenetTransform(name, img_size=256, crop_size=224, as_list=False, normalization=True):
    assert name in ["center", "random", "flip,crop", "flip,crop=5"]

    if name == "random":
        transform = [
            transforms.RandomResizedCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]
    # elif name == "flip,crop":
    #     pass
    else:
        transform = [
            transforms.Resize(img_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor()
        ]

    if normalization:
        if name == "tencrop":
            transform.append(transforms.Lambda(lambda crops: torch.stack([NORMALIZE_IMAGENET(crop) for crop in crops])))
        else:
            transform.append(NORMALIZE_IMAGENET)

    if as_list:
        return transform
    else:
        return transforms.Compose(transform)



def get_data_loader(img_size=256, crop_size=224, shuffle=False, nb_workers=8,
                    batch_size=64, distributed_sampler=False, dataset='imagenet', return_index=False,
                    data_path="", transform=None, split='valid'):
    """
    Get data loader over imagenet dataset.
    """
    assert dataset in DATASETS
    assert transform is not None

    pin_memory = (dataset == 'imagenet')

    if dataset.startswith("cifar"):
        transform = getCifarTransform(transform, img_size=img_size, crop_size=crop_size, normalization=True)
    else:
        transform = getImagenetTransform(transform, img_size=img_size, crop_size=crop_size, normalization=True)

    sampler = None

    if dataset == "cifar10":
        dataset = CIFAR10(root=DATASETS[dataset], name=split, transform=transform)
    elif dataset == "cifar100":
        dataset = CIFAR100(root=DATASETS[dataset][split], transform=transform)
    else:
        dataset = ImageFolder(root=DATASETS[dataset][split], transform=transform)


    # sampler
    if distributed_sampler:
        if sampler is None:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            sampler = DistributedSampler(sampler)

    # data loader
    if batch_size == -1:
        batch_size = len(dataset)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        num_workers=nb_workers,
        pin_memory=pin_memory,
        sampler=sampler
    )

    return data_loader, sampler
