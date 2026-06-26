import torch
import torch.nn as nn
import torch.nn.functional as F
    

# 1. SE Block 구현
class SEBlock(nn.Module):
    def __init__(self, c, r=16):    # r=16: excitation 단계에서 출력 차원을 줄이기 위한 변수
        super(SEBlock, self).__init__()
        # squeeze 단계:  (N, C, W, H) -> (N, C, 1, 1)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        # excitation 단계: (N, C, 1, 1) -> (N, C, 1, 1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),  # 활성화함수 ReLU
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input_filter):    # input_filter: 입력 feature map. (N, C, W, H)
        # 입력 feature map의 크기를 추출
        batch, channel, _, _ = input_filter.size()

        # squeeze 단계: (N, C, W, H) -> (N, C)
        se = self.squeeze(input_filter).view(batch, channel)

        # excitation 단계: (N, C, 1, 1) -> (N, C, 1, 1)
        # 각 채널의 중요도를 나타내는 3차원 텐서 생성
        se = self.excitation(se).view(batch, channel, 1, 1)

        # 최종 출력: 위에서 계산된 채널별 중요도를 입력 feature map에 반영
        return input_filter * se.expand_as(input_filter)


# 2. ASPP 구현
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.dilations = [6, 12, 18]  # Dilation rates

        self.aspp1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=self.dilations[0], dilation=self.dilations[0])
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.aspp2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=self.dilations[1], dilation=self.dilations[1])
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.aspp3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=self.dilations[2], dilation=self.dilations[2])
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.conv1x1 = nn.Conv2d(out_channels * 3, out_channels, kernel_size=1)  # Concat 후 차원 축소를 위한 1x1 Conv

    def forward(self, x):
        x1 = F.leaky_relu(self.bn1(self.aspp1(x)))
        x2 = F.leaky_relu(self.bn2(self.aspp2(x)))
        x3 = F.leaky_relu(self.bn3(self.aspp3(x)))

        x = torch.cat((x1, x2, x3), dim=1)  # Concatenate along the channel dimension
        x = self.conv1x1(x)

        return x


# 3. CBAM 구현
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
    

class ChannelAttention(nn.Module):
    def __init__(self, input_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.input_channels = input_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        #  https://github.com/luuuyi/CBAM.PyTorch/blob/master/model/resnet_cbam.py
        #  uses Convolutions instead of Linear
        self.MLP = nn.Sequential(
            Flatten(),
            nn.Linear(input_channels, input_channels // reduction_ratio),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(input_channels // reduction_ratio, input_channels)
        )

    def forward(self, x):
        # Take the input and apply average and max pooling
        avg_values = self.avg_pool(x)
        max_values = self.max_pool(x)
        out = self.MLP(avg_values) + self.MLP(max_values)
        scale = x * torch.sigmoid(out).unsqueeze(2).unsqueeze(3).expand_as(x)
        return scale


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        out = self.bn(out)
        scale = x * torch.sigmoid(out)
        return scale


class CBAM(nn.Module):
    def __init__(self, input_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(input_channels, reduction_ratio=reduction_ratio)
        self.spatial_att = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        out = self.channel_att(x)
        out = self.spatial_att(out)
        return out
    

# 4. Attention gate 구현
class AttBlock(nn.Module):
    """Attention block with learnable parameters"""

    def __init__(self, F_g, F_l, n_coefficients):
        """
        :param F_g: number of feature maps (channels) in previous layer
        :param F_l: number of feature maps in corresponding encoder layer, transferred via skip connection
        :param n_coefficients: number of learnable multi-dimensional attention coefficients
        """
        super(AttBlock, self).__init__()

        self.W_gate = nn.Sequential(
            nn.Conv2d(F_g, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(n_coefficients, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, gate, skip_connection):
        """
        :param gate: gating signal from previous layer
        :param skip_connection: activation from corresponding encoder layer
        :return: output activations
        """
        g1 = self.W_gate(gate)
        x1 = self.W_x(skip_connection)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = skip_connection * psi
        return out
    

# 5. Convolutional layer 구현
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    

class SEConv(nn.Module):
    """(conv => [BN] => ReLU) + SEBlock + residual"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv_se = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),  # LeakyReLU
            SEBlock(mid_channels)
        )
        self.residual = nn.Sequential(
             nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False),
             nn.LeakyReLU(negative_slope=0.2, inplace=True),  # LeakyReLU
        )

    def forward(self, x):
        return self.conv_se(x) + self.residual(x)
    

class ResDoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2 + residual"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.residual = nn.Sequential(
             nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False),
             nn.LeakyReLU(negative_slope=0.2, inplace=True)  # LeakyReLU
        )

    def forward(self, x):
        return self.double_conv(x) + self.residual(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)