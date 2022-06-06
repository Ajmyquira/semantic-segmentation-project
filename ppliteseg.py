import torch.nn as nn
from bisenet import BiSeNet

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

if __name__ == "__main__":
    backbone = BiSeNet('STDCNet813', 19)
    model = PPLiteSeg(19, backbone)