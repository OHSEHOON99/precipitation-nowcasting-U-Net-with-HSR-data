""" Full assembly of the parts to form the complete network """

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.drop1 = nn.Dropout(p=0.5)  # dropdout 구현완료
        self.down3 = (Down(256, 512))
        self.drop2 = nn.Dropout(p=0.5)  # dropdout 구현완료
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.drop3 = nn.Dropout(p=0.5)  # dropdout 구현완료
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x3_ = self.drop1(x3)    # dropdout 구현완료
        x4 = self.down3(x3_)
        x4_ = self.drop2(x4)    # dropdout 구현완료
        x5 = self.down4(x4_)
        x5_ = self.drop3(x5)    # dropdout 구현완료
        x = self.up1(x5_, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.drop1 = torch.utils.checkpoint(self.drop1)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.drop2 = torch.utils.checkpoint(self.drop2)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.drop3 = torch.utils.checkpoint(self.drop3)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)