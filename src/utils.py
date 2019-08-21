from __future__ import division
import torch
import torch.nn as nn
import torchvision.transforms as transforms


def boolify(x):
    assert x in ["yes", "no"]

    return x == "yes"


class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

def getTransform(data_aug):
    transfos = []
    if data_aug >= 1:
        transfos.append(transforms.RandomHorizontalFlip())
    if data_aug == 2:
        transfos.extend([transforms.Pad(1, fill=(128, 128, 128)),
                transforms.RandomCrop((32, 32))])
    elif data_aug == 3:
        transfos.extend([transforms.Pad(2, fill=(128, 128, 128)),
                transforms.RandomCrop((32, 32))])

    transfos.extend([
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    transform = transforms.Compose(transfos)

    return transform

def test(net, testloader, gpu):
    net = net.eval()
    correct = 0
    total = 0.0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            if gpu:
                inputs, labels = inputs.cuda(), labels.cuda()
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += float(labels.size(0))
            correct += float((predicted == labels).sum())

    return correct / total
