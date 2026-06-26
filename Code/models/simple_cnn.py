import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x

class EnsembleCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EnsembleCNN, self).__init__()
        self.layer1 = ConvBlock(in_channels, 48)
        self.layer2 = ConvBlock(48, 24)
        self.layer3 = ConvBlock(24, 12)
        self.layer4 = ConvBlock(12, 6)
        self.layer5 = ConvBlock(6, 3)
        self.outlayer = nn.Conv2d(3, out_channels, kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        out = self.outlayer(x)
        return out