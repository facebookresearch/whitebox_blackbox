from __future__ import division
import time
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
from os.path import join
from lib.utils import test, getTransform
from lib.datasets.cifar import CIFAR10
from lib import models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser()
parser.add_argument("--arch", type=str, choices=model_names)
parser.add_argument("--bs", type=int, default=4)
parser.add_argument("--dest", type=str, default=".")
parser.add_argument("--epochs", type=int, default=71)
parser.add_argument("--gpu", action='store_true', dest='gpu')
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--name", type=str, required=True)
parser.add_argument("--seed", type=int, default=0)
parser.set_defaults(gpu=True)
args = parser.parse_args()
print(args)

torch.cuda.manual_seed_all(args.seed)
transform = getTransform(0)

root_data = '/private/home/asablayrolles/data/cifar-dejalight'
nclasses = 10
trainset = CIFAR10(root=root_data, name=args.name,
                                        download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.bs,
                                          shuffle=True, num_workers=2)

testset = CIFAR10(root=root_data, name='val',
                                       download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.bs,
                                         shuffle=False, num_workers=2)

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
    best_val = val_accuracy

    if args.dest != "." and (epoch % 10 == 0 or epoch < 20):
        if args.gpu:
            net = net.cpu()
        print("Saving model")
        torch.save({
            'net': net.state_dict(),
            'args': args
        }, join(args.dest, "%s_epoch=%d.pth" % (args.arch, epoch)))
        if args.gpu:
            net = net.cuda()

print('Finished Training')
