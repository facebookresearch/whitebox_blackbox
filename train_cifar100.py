# -*- coding: utf-8 -*-
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import division
import _init_paths
import time
import torch
from torchvision.datasets import CIFAR100
import torch.optim as optim
import torch.nn as nn
import argparse
from os.path import join
from utils import test, getTransform, boolify
import models
import torchvision.transforms as transforms

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser()
parser.add_argument("--arch", type=str, choices=model_names, default="preactresnet18")
parser.add_argument("--bs", type=int, default=32)
parser.add_argument("--decay", type=boolify, default=False)
parser.add_argument("--dest", type=str, default=".")
parser.add_argument("--epochs", type=int, default=71)
parser.add_argument("--data-aug", type=int, default=0)
parser.add_argument("--gpu", action='store_true', dest='gpu')
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--workers", type=int, default=4)
parser.set_defaults(gpu=True)
args = parser.parse_args()
print(args)

torch.cuda.manual_seed_all(args.seed)

nclasses = 100
root_data = "/private/home/asablayrolles/data"

train_transform = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

val_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = CIFAR100(root=root_data, download=False, train=True, transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.bs, shuffle=True, num_workers=args.workers)

testset = CIFAR100(root=root_data, download=False, train=False, transform=val_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False, num_workers=args.workers)

net = models.__dict__[args.arch](nclasses)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
if args.gpu:
    net = net.cuda()
    criterion = criterion.cuda()

best_val = None
for epoch in range(args.epochs):
    start = time.time()
    net = net.train()
    avg_loss = 0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        if args.gpu:
            inputs, labels = inputs.cuda(), labels.cuda()

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        avg_loss += loss.data.item()

    avg_loss /= len(trainloader)
    net = net.eval()
    val_accuracy = test(net, testloader, args.gpu)

    print("Epoch %d, took %.2f, loss=%.2f, train acc=%.2f, val acc=%.2f" % (
        epoch,
        time.time() - start,
        avg_loss,
        test(net, trainloader, args.gpu),
        val_accuracy
    ))

    if best_val is None:
        best_val = val_accuracy
    if val_accuracy < best_val:
        if args.decay:
            print("Dividing learning rate by 1.1")
            args.lr /= 2
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)

    best_val = val_accuracy

    if args.dest != "." and (epoch % 10 == 0 or epoch < 20):
        if args.gpu:
            net = net.cpu()
        print "Saving model"
        torch.save({'net': net.state_dict(), 'args': args}, join(args.dest, "%s_epoch=%d.pth" % (args.arch, epoch)))
        if args.gpu:
            net = net.cuda()

print('Finished Training')
