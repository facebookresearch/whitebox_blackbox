# Borrowed from https://github.com/kuangliu/pytorch-cifar/blob/master/models/preact_resnet.py

'''Pre-activation ResNet in PyTorch.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, branches=1, expand=False, subsample=1):
        super(PreActBlock, self).__init__()

        if expand:
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.conv1 = nn.Conv2d(in_planes, planes * branches // subsample, kernel_size=3, stride=stride, padding=1, bias=False)
        else:
            self.bn1 = nn.BatchNorm2d(in_planes * branches // subsample)
            self.conv1 = nn.Conv2d(in_planes * branches // subsample, planes * branches // subsample, kernel_size=3, stride=stride, padding=1, bias=False, groups=branches)

        self.bn2 = nn.BatchNorm2d(planes * branches // subsample)
        self.conv2 = nn.Conv2d(planes * branches // subsample, planes * branches // subsample, kernel_size=3, stride=1, padding=1, bias=False, groups=branches)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes*branches // subsample, self.expansion*planes*branches // subsample, kernel_size=1, stride=stride, bias=False, groups=branches)
            )
        if expand:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes*branches // subsample, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, branches=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes * branches)
        self.conv1 = nn.Conv2d(in_planes * branches, planes * branches, kernel_size=1, bias=False, groups=branches)
        self.bn2 = nn.BatchNorm2d(planes * branches)
        self.conv2 = nn.Conv2d(planes * branches, planes * branches, kernel_size=3, stride=stride, padding=1, bias=False, groups=branches)
        self.bn3 = nn.BatchNorm2d(planes * branches)
        self.conv3 = nn.Conv2d(planes * branches, self.expansion*planes * branches, kernel_size=1, bias=False, groups=branches)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes * branches, self.expansion*planes * branches, kernel_size=1, stride=stride, bias=False, groups=branches)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class Preactresnet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, branches=1, subsample=1, branch_level=4):
        super(Preactresnet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        if branch_level == 3:
            s = subsample
            self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, branches=branches, subsample=s, expand=True)
        else:
            self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, branches=branches, subsample=subsample, expand=(branch_level == 4))

        self.branches = branches
        self.linears = branches * [nn.Linear(512*block.expansion // subsample, num_classes // branches)]
        for i in range(self.branches):
            self.__setattr__("linear_%d" % i, self.linears[i])
        # self.linear = nn.Linear(512*block.expansion*branches, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, branches=1, subsample=1, expand=False):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        isfirst = True
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride=stride, branches=branches, expand=(expand and isfirst), subsample=subsample))
            isfirst = False
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), self.branches, -1)

        out = [self.linears[i](out[:, i, :]) for i in range(self.branches)]
        out = [t.view(t.size(0), t.size(1), 1) for t in out]
        out = torch.cat(out, dim=2)

        return out.view(out.size(0), -1)


def preactresnet18(nclasses, nonlinearity='relu', branches=1, subsample=1, branch_level=4, **kwargs):
    assert nonlinearity == 'relu'
    assert branch_level in [3, 4]
    return Preactresnet(PreActBlock, [2,2,2,2], num_classes=nclasses, branches=branches, subsample=subsample, branch_level=branch_level)

def preactresnet34(nclasses, nonlinearity='relu', branches=1, **kwargs):
    assert nonlinearity == 'relu'
    assert False, "Need to add subsample() to this code"
    return Preactresnet(PreActBlock, [3,4,6,3], num_classes=nclasses, branches=branches)

def preactresnet50(nclasses, nonlinearity='relu', branches=1, **kwargs):
    assert nonlinearity == 'relu'
    assert branches == 1
    assert False, "Need to add subsample() to this code"
    return Preactresnet(PreActBottleneck, [3,4,6,3], num_classes=nclasses, branches=branches)

def preactresnet101(nclasses, nonlinearity='relu', branches=1, **kwargs):
    assert nonlinearity == 'relu'
    assert branches == 1
    assert False, "Need to add subsample() to this code"
    return Preactresnet(PreActBottleneck, [3,4,23,3], num_classes=nclasses, branches=branches)

def preactresnet152(nclasses, nonlinearity='relu', branches=1, **kwargs):
    assert nonlinearity == 'relu'
    assert branches == 1
    assert False, "Need to add subsample() to this code"
    return Preactresnet(PreActBottleneck, [3,8,36,3], num_classes=nclasses, branches=branches)
