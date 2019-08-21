from __future__ import division
import numpy as np
import src.models as models
import torch
from os.path import join
# from src.datasets.cifar import CIFAR10
from src.dataset import get_data_loader
import torch.nn as nn
from itertools import chain
from os.path import dirname
import argparse
import time



# def get_parser():
#     """
#     Generate a parameters parser.
#     """
#     # parse parameters
#     parser = argparse.ArgumentParser(description='Language transfer')
#
#     parser.add_argument("--nb_workers", type=int, default=10,
#                         help="Number of workers")
#
#     # dataset
#     parser.add_argument("--dataset", choices=["imagenet", "cifar10", "cifar100"], default="imagenet",
#                         help="Dataset")
#     parser.add_argument("--transform", choices=["center", "random", "flip", "flip,crop=5", "flip,crop"], required=True,
#                         help="Data augmentation transformation")
#     parser.add_argument("--split_train", type=str, default="train",
#                         help="Which splits corresponds to the training set")
#     # parser.add_argument("--crop_size", type=int, default=224,
#     #                     help="Crop size")
#     # parser.add_argument("--img_size", type=int, default=256,
#     #                     help="Img resize")
#
#     # model type
#     parser.add_argument("--architecture", type=str, default="resnet18",
#                         help="Architecture (resnet18, resnet34, resnet50, resnet101, resnet152)")
#     # parser.add_argument("--pretrained", type=bool_flag, default=False,
#     #                     help="Use a pretrained model")
#
#     # training parameters
#     parser.add_argument("--optimizer", type=str, default="sgd,lr=0.1-0.01-0.001,momentum=0.9,weight_decay=0.0001",
#                         help="Optimizer (SGD / RMSprop / Adam, etc.)")
#     parser.add_argument("--batch_size", type=int, default=4,
#                         help="Number of sentences per batch")
#     parser.add_argument("--epochs", type=int, default=50,
#                         help="Number of epochs")
#     parser.add_argument("--validation_metrics", type=str, default="",
#                         help="Validation metrics")
#     parser.add_argument("--stopping_criterion", type=str, default="",
#                         help="Stopping criterion, and number of non-increase before stopping the experiment")
#
#     # evaluation
#     parser.add_argument("--eval_only", type=bool_flag, default=False,
#                         help="Only run evaluations")
#
#     # debug
#     parser.add_argument("--debug_train", type=bool_flag, default=False,
#                         help="Use valid sets for train sets (faster loading)")
#     parser.add_argument("--debug_slurm", type=bool_flag, default=False,
#                         help="Debug from a SLURM job")
#     parser.add_argument("--debug", help="Enable all debug flags",
#                         action="store_true")
#
#     # multi-gpu / multi-node
#     parser.add_argument("--local_rank", type=int, default=-1,
#                         help="Multi-GPU - Local rank")
#     parser.add_argument("--master_port", type=int, default=-1,
#                         help="Master port (for multi-node SLURM jobs)")
#
#     return parser
#


def getLosses(img, lbl, net):
    criterion = nn.CrossEntropyLoss(reduction='none')
    net = net.eval()
    with torch.no_grad():
        out = net(img)
        loss = criterion(out, lbl)

    return loss.numpy()


def get_data(split):
    params = argparse.Namespace()
    params.crop_size = 32
    params.dataset = "cifar10"
    params.img_size = 40
    params.multi_gpu = False
    params.nb_workers = 2
    params.transform = 'center'


    train_data_loader, _ = get_data_loader(
        img_size=params.img_size,
        crop_size=params.crop_size,
        shuffle=False,
        batch_size=-1,
        nb_workers=params.nb_workers,
        distributed_sampler=params.multi_gpu,
        dataset=params.dataset,
        transform=params.transform,
        split=split
    )

    img, lbl = next(iter(train_data_loader))
    print(img.size())
    print(lbl.size())

    return img, lbl


#root_data = "/private/home/asablayrolles/data/cifar-dejalight/idx_public_%d.npy"
root_data = "/private/home/asablayrolles/data/cifar-dejalight2/idx_public_%d.npy"
n_img = 60000 # 50000


indices = [np.load(root_data % i) for i in range(30)]
public_indices = set(chain.from_iterable([set(t) for t in indices]))
public_indices = np.array(list(public_indices))

seen = np.zeros((30, n_img), dtype=bool)

for i_idx, idx in enumerate(indices):
    seen[i_idx][idx] = 1


seen = seen.T[public_indices]

ckpt_root = "/checkpoint/asablayrolles/dejalight_cifar/"
epoch = 50

nets = []
for i in range(30):
    net = models.smallnet(10)
    #state_dict = torch.load(join(ckpt_root, "train_estim_all/_name=public_%d/smallnet_epoch=%d.pth" % (i, epoch)))
    state_dict = torch.load(join(ckpt_root, "train_estim_smallbatch_final/_name=public_%d/smallnet_epoch=%d.pth" % (i, epoch)))
    net.load_state_dict(state_dict["net"])
    nets.append(net)

img, lbl = get_data("public")

losses = []
start = time.time()
for i_net, net in enumerate(nets):
    losses.append(getLosses(img, lbl, net))
    eta = (len(nets) - (i_net + 1)) * (time.time() - start) / (i_net + 1)
    if i_net % 10 == 0:
        print("ETA: %.2f seconds" % eta)


losses = np.vstack(losses).T
losses_heldout = losses[:, -1]  # (n_images, )
losses = losses[:, :-1]         # (n_images, n_models - 1)

seen_heldout = seen[:, -1]  # (n_images, )
seen = seen[:, :-1]         # (n_images, n_models - 1)

logloss = np.log(1e-8 + losses)

best_taus = []
smoothed_taus = []

print("There are %d elements" % logloss.shape[0])
for id_element in range(logloss.shape[0]):
    loss_seen = logloss[id_element][seen[id_element]]
    loss_unseen = logloss[id_element][np.logical_not(seen[id_element])]

    num_pos = np.sum(seen[id_element])
    num_neg = seen[id_element].shape[0] - num_pos

    values = np.concatenate([loss_seen, loss_unseen])
    sugar = np.concatenate([np.ones((num_pos)) / num_pos, - np.ones(num_neg) / num_neg])
    order = np.argsort(values)

    best_pos = np.argmax(np.cumsum(sugar[order]))
    best_tau = values[order[best_pos]]

    if best_pos >= 2 and best_pos < order.shape[0] - 2:
        smoothed_tau = np.mean([values[order[best_pos-2:best_pos]], values[order[best_pos+1:best_pos+3]]])
    else:
        smoothed_tau = best_tau

    best_taus.append(best_tau)
    smoothed_taus.append(smoothed_tau)

best_taus = np.array(best_taus)
smoothed_taus = np.array(smoothed_taus)

print("Smoothed tau", np.mean(np.exp(smoothed_taus)))
print("Best tau", np.mean(np.exp(best_taus)))

print("Perf", np.mean(
    np.logical_xor(
        losses_heldout - np.exp(smoothed_taus) >= 0,
        seen_heldout
    )
))


net_private = models.smallnet(10)
state_dict = torch.load(join(ckpt_root, "train_private_smallbatch_final/expe/smallnet_epoch=%d.pth" % epoch))
net_private.load_state_dict(state_dict["net"])

img_private, lbl_private = get_data("private")
losses_private = getLosses(img_private, lbl_private, net_private)

img_test, lbl_test = get_data("test")
losses_test = getLosses(img_test, lbl_test, net_private)

assert len(losses_test) == len(losses_private)
all_losses = np.append(losses_test, losses_private)
membership = np.append(np.zeros((len(losses_test))), np.ones((len(losses_private))))

print("Perf", np.mean(
    np.logical_xor(
        all_losses - np.mean(np.exp(smoothed_taus)) >= 0,
        membership
    )
))

import ipdb; ipdb.set_trace()
