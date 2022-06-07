import torch.nn as nn
import torch.nn.functional as F
from stdcseg import STDCNet

class ConvBNReLU(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding='same'):
        super().__init__()

        self._conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding)
        self._batch_norm = nn.BatchNorm2d(out_channels)
        self._relu = nn.ReLU()

    def forward(self, x):
        x = self._conv(x)
        x = self._batch_norm(x)
        x = self._relu(x)
        return x

class PPContextModule(nn.Module):
    """
    Simple Context module.
    Args:
        in_channels (int): The number of input channels to pyramid pooling module.
        inter_channels (int): The number of inter channels to pyramid pooling module.
        out_channels (int): The number of output channels after pyramid pooling module.
        bin_sizes (tuple, optional): The out size of pooled feature maps. Default: (1, 3).
        align_corners (bool): An argument of F.interpolate. It should be set to False
            when the output size of feature is even, e.g. 1024x512, otherwise it is True, e.g. 769x769.
    """

    def __init__(self,
                 in_channels,
                 inter_channels,
                 out_channels,
                 bin_sizes,
                 align_corners=False):
        super().__init__()

        self.stages = nn.ModuleList([
            self._make_stage(in_channels, inter_channels, size)
            for size in bin_sizes
        ])

        self.conv_out = ConvBNReLU(
            in_channels=inter_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1)

        self.align_corners = align_corners

    def _make_stage(self, in_channels, out_channels, size):
        prior = nn.AdaptiveAvgPool2d(output_size=size)
        conv = ConvBNReLU(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        return nn.Sequential(prior, conv)

    def forward(self, input):
        out = None
        input_shape = input.shape[2:]

        for stage in self.stages:
            x = stage(input)
            x = F.interpolate(
                x,
                input_shape,
                mode='bilinear',
                align_corners=self.align_corners)
            if out is None:
                out = x
            else:
                out += x

        out = self.conv_out(out)
        return out

class PPLiteSegHead(nn.Module):
    """
    The head of PPLiteSeg.
    Args:
        backbone_out_chs (List(Tensor)): The channels of output tensors in the backbone.
        arm_out_chs (List(int)): The out channels of each arm module.
        cm_bin_sizes (List(int)): The bin size of context module.
        cm_out_ch (int): The output channel of the last context module.
        arm_type (str): The type of attention refinement module.
        resize_mode (str): The resize mode for the upsampling operation in decoder.
    """
    def __init__(self, backbone_out_chs, arm_out_chs, cm_bin_sizes, cm_out_ch,
                 arm_type, resize_mode):
        super().__init__()

        self.cm = PPContextModule(backbone_out_chs[-1], cm_out_ch, cm_out_ch,
                                  cm_bin_sizes)

        # assert hasattr(layers, arm_type)

class PPLiteSeg(nn.Module):
    def __init__(self,
                 num_classes,
                 backbone,
                 backbone_indices=[2, 3, 4],
                 arm_type='UAFM_SpAtten',
                 cm_bin_sizes=[1, 2, 4],
                 cm_bin_ch=128,
                 arm_out_chs=[64, 96, 128],
                 seg_head_inter_chs=[64, 64, 64],
                 resize_mode='bilinear',
                 pretrained=None):
        super().__init__()

        # backbone
        assert hasattr(backbone, 'feat_channels'), \
            "The backbone should has feat_channels."
        assert len(backbone.feat_channels) >= len(backbone_indices), \
            f"The length of input backbone_indices ({len(backbone_indices)}) should not be" \
            f"greater than the length of feat_channels ({len(backbone.feat_channels)})."
        assert len(backbone.feat_channels) > max(backbone_indices), \
            f"The max value ({len(backbone_indices)}) of backbone_indices should be" \
            f"less than the length of feat_channels ({len(backbone.feat_channels)})."
        self.backbone = backbone

        assert len(backbone_indices) > 1, "The lenght of backbone_indices " \
            "should be greater than 1"
        self.backbone_indices = backbone_indices # [..., x16_id, x32_id]
        backbone_out_chs = [backbone.feat_channels[i] for i in backbone_indices]

        # head
        if len(arm_out_chs) == 1:
            arm_out_chs = arm_out_chs * len(backbone_indices)
        assert len(arm_out_chs) == len(backbone_indices), "The length of " \
            "arm_out_chs and backbone_indices should be equal"

        # self.ppseg_head = PPLiteSegHead()

if __name__ == "__main__":
    # STDC1
    backbone = STDCNet(base=64, layers=[2, 2, 2])
    model = PPLiteSeg(19, backbone)