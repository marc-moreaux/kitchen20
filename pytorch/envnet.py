"""
 Implementation of EnvNet [Tokozume and Harada, 2017]
 opt.fs = 16000
 opt.inputLength = 24014
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize,
                 stride=1,
                 pad=0,
                 bias=False,
                 use_bn=True):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=ksize,
                              stride=stride,
                              padding=pad,
                              bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        return x


def _init_weights(m):
    if type(m) == nn.Conv2d:
        nn.init.xavier_normal_(m.weight)


class EnvNet(nn.Module):
    def __init__(self, n_classes, use_GAP=False):
        super(EnvNet, self).__init__()
        self.use_GAP = use_GAP

        # Conv layers
        self.conv1 = ConvBNReLU(1, 40, (1, 8))
        self.conv2 = ConvBNReLU(40, 40, (1, 8))
        self.conv3 = ConvBNReLU(1, 50, (8, 13))
        self.conv4 = ConvBNReLU(50, 50, (1, 5))
        
        # FC layers
        self.fc5 = nn.Linear(50 * 11 * 26, 4096)
        self.fc6 = nn.Linear(4096, 4096)
        self.fc7 = nn.Linear(4096, n_classes)

        # GAP layers
        self.conv5 = ConvBNReLU(50, 50, (3, 3), use_bn=False)
        self.convGAP = ConvBNReLU(50, n_classes, (1, 1))

        # Dropout layers
        self.dropout1 = nn.Dropout(.5)
        self.dropout2 = nn.Dropout(.5)

        # Init weights
        self.apply(_init_weights)

    def forward(self, x):
        h = self.conv1(x)
        h = self.conv2(h)
        h = nn.MaxPool2d(kernel_size=(1, 160))(h)
        h = h.permute(0, 2, 1, 3)
        #h = F.swapaxes(h, 1, 2)

        h = self.conv3(h)
        h3 = nn.MaxPool2d(3)(h)
        h = self.conv4(h3)
        h = nn.MaxPool2d((1, 3))(h)

        if self.use_GAP:
            h = self.conv5(h)
            h = self.convGAP(h)
            self.maps = h
            h = F.adaptive_avg_pool2d(h, (1, 1))
            return h
        else:
            h = self.dropout1(F.relu(self.fc5(h)))
            h = self.dropout2(F.relu(self.fc6(h)))
            return self.fc7(h)


if __name__ == '__main__':
    envnet = EnvNet(10, True)
    def init_weights(m):
        if type(m) == nn.Conv2d:
            nn.init.xavier_normal_(m.weight)

    envnet.apply(init_weights)
    inputs = torch.rand((1,1,1,16000))

    envnet.train()
    print(envnet(inputs).tolist())

    envnet.eval()
    print(envnet(inputs).tolist())
