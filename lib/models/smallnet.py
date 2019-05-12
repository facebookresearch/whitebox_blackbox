import torch.nn as nn

class SmallNet(nn.Module):
    def __init__(self, nclasses, nfeat=32):
        super(SmallNet, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, nfeat, kernel_size=5, padding=2),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Conv2d(nfeat, nfeat, kernel_size=5, padding=2),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=4, stride=4))

        self.fc = [nn.Linear(2*2*nfeat, 128), nn.Tanh(), nn.Linear(128, nclasses)]
        self.fc = nn.Sequential(*self.fc)


    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def smallnet(nclasses=10):
    return SmallNet(nclasses)
