import torch.nn as nn
import torch.nn.functional as F
from operation import ConvBNReLU, ConvBN
from functions import avg_max_reduce_channel

class UAFM(nn.Module):
    """
    The base of Unified Attention Fusion Module.
    Args:
        x_ch (int): The channel of x tensor, which is the low level feature.
        y_ch (int): The channel of y tensor, which is the high level feature.
        out_ch (int): The channel of output tensor.
        ksize (int, optional): The kernel size of the conv for x tensor. Default: 3.
        resize_mode (str, optional): The resize model in unsampling y tensor. Default: bilinear.
    """

    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__()

        self.conv_x = ConvBNReLU(
            x_ch, y_ch, kernel_size=ksize, padding=ksize // 2, bias=False)
        self.conv_out = ConvBNReLU(
            y_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.resize_mode = resize_mode

    def check(self, x, y):
        assert x.ndim == 4 and y.dim == 4
        x_h, x_w = x.shape[2:]
        y_h, y_w = y.shape[2:]
        assert x_h >= y_h and x_w >= y_w

    def prepare(self, x, y):
        x = self.prepare_x(x, y)
        y = self.prepare_y(x, y)
        return x, y

    def prepare_x(self, x, y):
        x = self.conv_x(x)
        return x
    def prepare_y(self, x, y):
        # Upsampling
        y_up = F.interpolate(y, x.shape[2:], mode=self.resize_mode)
        return y_up

    def fuse(self, x, y):
        out = x + y
        out = self.conv_out(out)
        return out

    def forward(self, x, y):
        """
        Args:
            x (Tensor): The low level feature.
            y (Tensor): The high level feature.
        """
        self.check(x, y)
        x, y = self.prepare(x, y)
        out = self.fuse(x, y)
        return out

class UAFM_SpAtten(UAFM):
    """
    The UAFM with spatial attention, which uses mean and max values.
    Args:
        x_ch (int): The channel of x tensor, which is the low level feature.
        y_ch (int): The channel of y tensor, which is the high level feature.
        out_ch (int): The channel of output tensor.
        ksize (int, optional): The kernel size of the conv for x tensor. Default: 3.
        resize_mode (str, optional): The resize model in unsampling y tensor. Default: bilinear.
    """

    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_ch, y_ch, out_ch, ksize, resize_mode)

        self.conv_xy_atten = nn.Sequential(
            ConvBNReLU(4, 2, kernel_size=3, padding=1, bias=False),
            ConvBN(2, 1, kernel_size=3, padding=1, bias=False))

    def fuse(self, x, y):
        """
        Args:
            x (Tensor): The low level feature.
            y (Tensor): The high level feature.
        """
        atten = avg_max_reduce_channel([x, y])
        atten = F.sigmoid(self.conv_xy_atten(atten))

        out = x * atten + y * (1 - atten)
        out = self.conv_out(out)
        return out