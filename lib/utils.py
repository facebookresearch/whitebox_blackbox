from __future__ import division
import numpy as np
import torch
import torch.nn as nn
import time
from torchvision.models import VGG
import torch.nn.functional as F
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

def train_net(net, trainloader, optimizer, criterion, gpu, verbose=False):
    avg_loss = 0
    start = time.time()
    net = net.train()
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        if gpu:
            inputs, labels = inputs.cuda(), labels.cuda()
        inputs, labels = Variable(inputs), Variable(labels)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        avg_loss += loss.data[0]
        if verbose:
            print("Loss so far: %.2f" % (avg_loss / (i + 1)))
        loss.backward()
        optimizer.step()

    avg_loss /= len(trainloader)

    return avg_loss, (time.time() - start)


def train_net_cl(net, trainloader, optimizer, criterion, gpu):
    avg_loss = 0
    correct, total = 0, 0
    start = time.time()
    net = net.train()
    confusion_matrix = None
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        if gpu:
            inputs, labels = inputs.cuda(), labels.cuda()
        inputs, labels = Variable(inputs), Variable(labels)

        optimizer.zero_grad()
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.data).sum()
        if confusion_matrix is None:
            confusion_matrix = np.zeros((outputs.size(1), outputs.size(1)))
        for i in range(labels.size(0)):
            confusion_matrix[predicted[i], labels.data[i]] += 1
        loss = criterion(outputs, labels)
        avg_loss += loss.data[0]
        loss.backward()
        optimizer.step()

    avg_loss /= len(trainloader)

    return avg_loss, correct / total, (time.time() - start), confusion_matrix

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


def test_binary(net, testloader, gpu, rare_label=0):
    net = net.eval()
    preds = []
    gt = []
    confs = []
    for data in testloader:
        inputs, labels = data
        if gpu:
            inputs, labels = inputs.cuda(), labels.cuda()
        outputs = net(Variable(inputs, volatile=True))
        _, predicted = torch.max(outputs.data, 1)
        preds.append(predicted.cpu().numpy())
        gt.append(labels.cpu().numpy())
        confs.append(F.softmax(outputs)[1])

    gt = np.concatenate(gt, axis=0)
    preds = np.concatenate(preds, axis=0)
    confs = np.concatenate(confs, axis=0)
    # Check that the label is indeed rare
    # assert np.sum(gt == rare_label) < np.sum(gt != rare_label)

    prec = np.mean(gt[preds == rare_label] == rare_label)
    recall = np.sum(np.logical_and(preds == rare_label, gt == rare_label)) / np.sum(gt == rare_label)

    return prec, recall, np.mean(gt == preds)


def test_loss(net, loader, criterion, gpu):
    avg_loss = 0
    net = net.eval()
    for i, data in enumerate(loader, 0):
        inputs, labels = data
        if gpu:
            inputs, labels = inputs.cuda(), labels.cuda()
        inputs, labels = Variable(inputs, volatile=True), Variable(labels)

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        avg_loss += loss.data[0]

    avg_loss /= len(loader)

    return avg_loss

class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        return torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)


def cut(net, config):
    if config.startswith("fc"):
        numlayer = int(config[2:])
        layers = net.fc._modules.values()
        net.fc = nn.Sequential(*layers[:numlayer])
    elif config.startswith("conv-avg"):
        numlayer = int(config[8:])
        layers = net.conv._modules.values()
        net = nn.Sequential(nn.Sequential(*layers[:numlayer]), GlobalAvgPool2d())
    elif config.startswith("conv"):
        numlayer = int(config[4:])
        net.fc = nn.Sequential()
        layers = net.conv._modules.values()
        net.conv = nn.Sequential(*layers[:numlayer])
    else:
        raise NotImplementedError("Config should be fcX or convX or conv-avgX, it is %s" % config)

    return net

def splitVGG(net, config):
    firstNet, secondNet = [], []
    if config.startswith("features"):
        numlayer = int(config[9:])
        layers = net.features._modules.values()

        assert numlayer >= 1
        assert numlayer <= len(layers)

        firstNet.extend(layers[:numlayer])
        secondNet.extend(layers[numlayer:])
        secondNet.append(FlattenLayer())
        secondNet.append(net.classifier)
    elif config.startswith("classifier"):
        numlayer = int(config[11:])
        layers = net.classifier._modules.values()

        assert numlayer >= 1
        assert numlayer < len(layers)

        firstNet.append(net.features)
        firstNet.append(FlattenLayer())
        firstNet.extend(layers[:numlayer])

        secondNet.extend(layers[numlayer:])
    else:
        raise NotImplementedError("Config should be fcX or convX, it is %s" % config)

    firstNet, secondNet = nn.Sequential(*firstNet), nn.Sequential(*secondNet)

    return firstNet, secondNet

def splitNet(net, config):
    """ Return two networks: the one until layer config, and the one after layer
    config"""


    if net.__class__ == VGG:
        return splitVGG(net, config)

    if config.startswith("fc"):
        numlayer = int(config[2:])
        layers = net.fc._modules.values()
        net.fc = nn.Sequential(*layers[:numlayer])

        firstPart = net
        secondPart = nn.Sequential(*layers[numlayer:])
    elif config.startswith("conv"):
        numlayer = int(config[4:])
        layers = net.conv._modules.values()
        net.conv = nn.Sequential(*layers[numlayer:])

        firstPart = nn.Sequential(*layers[:numlayer])
        secondPart = net
    else:
        raise NotImplementedError("Config should be fcX or convX, it is %s" % config)

    return firstPart, secondPart


def extract(loader, istr, net, gpu, offset, features, istrain, classes, ntotal, postprocess=lambda x:x, force_training_mode=False, verbose=False, stop=False):
    shap = None
    import time
    start = time.time()

    assert force_training_mode or (not net.training)
    for i, data in enumerate(loader, 0):
        inputs, labels = data
        if inputs.dim() == 5:
            _, _, c, h, w = inputs.size()
            inputs = inputs.view(-1, c, h, w)
            labels = labels.view(-1)

        sz = inputs.size(0)
        if gpu:
            inputs = inputs.cuda()
        inputs = Variable(inputs, volatile=True)

        activations = net(inputs)
        if features is None:
            shap = activations.shape

        activations = postprocess(activations.data.cpu().numpy())

        if features is None:
            features = np.zeros((ntotal, activations.shape[1]), dtype=np.float32)
            print("Allocating %.2f GB" % (features.nbytes / 1e9))

        features[offset:(offset+sz)] = activations
        istrain[offset:(offset+sz)] = istr
        classes[offset:(offset+sz)] = labels.numpy()
        offset += sz

        if stop and offset >= ntotal:
            break

        if verbose and (i % 20 == 19):
            speed = (time.time() - start) / ((i + 1) * inputs.size(0))
            print("ETA: %.2f" % (speed * (features.shape[0] - offset)))
            print("offset: %d" % offset)

    return offset, features, shap

def splitResnet(net, layernum, levelnum, nclasses=-1):
    numlayers = 1 + max([i for i in range(10) if hasattr(net, "layer%d" % i)])
    assert layernum <= numlayers and layernum >= 1
    # assert levelnum <= 4 and levelnum >= 1

    firstNet, secondNet = [], []
    firstNet.append(net.conv1)
    firstNet.append(net.bn1)
    firstNet.append(net.relu)
    if hasattr(net, "maxpool"):
        firstNet.append(net.maxpool)

    for layer in range(1, numlayers):
        modules = net.__getattr__("layer%d" % layer)._modules.values()
        if layer < layernum:
            firstNet.extend(modules)
        elif layer > layernum:
            secondNet.extend(modules)
        else:
            firstNet.extend(modules[:levelnum])
            secondNet.extend(modules[levelnum:])

    secondNet.append(net.avgpool)
    secondNet.append(FlattenLayer())
    if nclasses != -1:
        secondNet.append(nn.Linear(net.fc.weight.size(1), nclasses).cuda())
    else:
        secondNet.append(net.fc)

    firstNet = nn.Sequential(*firstNet)
    secondNet = nn.Sequential(*secondNet)

    return firstNet, secondNet
