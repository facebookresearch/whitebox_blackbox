# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
import torch.nn as nn

class SmallNet(nn.Module):
    def __init__(self, num_classes, num_channels, num_fc, maxpool_size, kernel_size, non_linearity):
        super(SmallNet, self).__init__()

        if non_linearity == "tanh":
            nonlinear_layer = nn.Tanh()
        elif non_linearity == "relu":
            nonlinear_layer = nn.ReLU()
        else:
            raise NotImplementedError()

        # self.conv = nn.Sequential(
        #     nn.Conv2d(3, num_channels, kernel_size=3, padding=1),
        #     nonlinear_layer,
        #     nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
        #     nonlinear_layer,
        #     nn.MaxPool2d(kernel_size=maxpool_size),
        #     nn.Conv2d(num_channels, 2 * num_channels, kernel_size=3, padding=1),
        #     nonlinear_layer,
        #     nn.Conv2d(2 * num_channels, 2 * num_channels, kernel_size=3, padding=1),
        #     nonlinear_layer,
        #     nn.MaxPool2d(kernel_size=maxpool_size)
        # )
        self.conv = nn.Sequential(
            nn.Conv2d(3, num_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
            nonlinear_layer,
            nn.MaxPool2d(kernel_size=maxpool_size),
            nn.Conv2d(num_channels, 2 * num_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
            nonlinear_layer,
            nn.MaxPool2d(kernel_size=maxpool_size)
        )

        self.fc = [nn.Linear(32 * 32 // (maxpool_size**4) * 2 * num_channels, num_fc), nonlinear_layer, nn.Linear(num_fc, num_classes)]
        self.fc = nn.Sequential(*self.fc)


    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def smallnet(num_classes, num_channels, num_fc, maxpool_size, kernel_size, non_linearity):
    return SmallNet(num_classes, num_channels, num_fc, maxpool_size, kernel_size, non_linearity)
