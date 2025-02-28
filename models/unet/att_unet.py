""" Full assembly of the parts to form the complete network """
"""bilinear is not being implemented yet"""

from .model_parts import *


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(UpConv, self).__init__()

        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(inplace=True)
            )

        else:
            self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2),
            nn.Conv2d(in_channels // 2, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
            )

    def forward(self, x):
        x = self.up(x)
        return x


class Up(nn.Module):
    """Upscaling then DoubleConv with AttBlock"""

    def __init__(self, in_channels, out_channels, n_coefficients, bilinear=True):
        super().__init__()
        if bilinear:
            self.up1 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(in_channels // 2),
                nn.LeakyReLU(inplace=True)
            )
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

        else:
            self.up1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2),
            nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.LeakyReLU(inplace=True)
            )
            self.conv = DoubleConv(in_channels, out_channels)

        self.att = AttBlock(out_channels, out_channels, n_coefficients)

    def forward(self, d, s):
        x1 = self.up1(d)
        x2 = self.att(gate=x1, skip_connection=s)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then DoubleConv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
    


class AttUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(AttUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # encoder
        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))

        # decoder
        self.up4 = (Up(1024, 512 // factor, 256))
        self.up3 = (Up(512, 256 // factor, 128))
        self.up2 = (Up(256, 128 // factor, 64))
        self.up1 = (Up(128, 64 // factor, 32))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        """
        e : encoder layers
        d : decoder layers
        s : skip-connections from encoder layers to decoder layers
        """
        # encoder
        e1 = self.inc(x)
        e2 = self.down1(e1)
        e3 = self.down2(e2)
        e4 = self.down3(e3)
        e5 = self.down4(e4)

        # decoder
        d4 = self.up4(e5, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)
        logits = self.outc(d1)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)