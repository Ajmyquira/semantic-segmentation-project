import torch
import torch.nn as nn
import math

class ConvBNRelu(nn.Module):
    def __init__(self, in_planes, out_planes, kernel=3, stride=1):
        super(ConvBNRelu, self).__init__()
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel,
            stride=stride,
            padding=kernel // 2,
            bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return out

class CatBottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, block_num=3, stride=4):
        super(CatBottleneck, self).__init__()
        assert block_num > 1, print("Block number should be larger than 1.")
        self.conv_list = nn.ModuleList()
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2d(
                    out_planes // 2,
                    out_planes // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=out_planes // 2,
                    bias=False),
                nn.BatchNorm2d(out_planes // 2), )
            self.skip = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(
                    ConvBNRelu(in_planes, out_planes // 2, kernel=1))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(
                    ConvBNRelu(out_planes // 2, out_planes // 2, stride=stride))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(
                    ConvBNRelu(out_planes // 2, out_planes // 4, stride=stride))
            elif idx < block_num - 1:
                self.conv_list.append(
                    ConvBNRelu(out_planes // int(math.pow(2, idx)), out_planes
                               // int(math.pow(2, idx + 1))))
            else:
                self.conv_list.append(
                    ConvBNRelu(out_planes // int(math.pow(2, idx)), out_planes
                               // int(math.pow(2, idx))))
    def forward(self, x):
        out_list = []
        out1 = self.conv_list[0](x)
        for idx, conv in enumerate(self.conv_list[1:]):
            if idx == 0:
                if self.stride == 2:
                    out = conv(self.avd_layer(out1))
                else:
                    out = conv(out1)
            else:
                out = conv(out)
            out_list.append(out)
        if self.stride == 2:
            out1 = self.skip(out1)
        out_list.insert(0, out1)
        out = torch.cat(out_list, dim=1)
        return out

class AddBottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, block_num=3, stride=1):
        super(AddBottleneck, self).__init__()
        assert block_num > 1, print("Block number should be larger than 1.")
        self.conv_list = nn.ModuleList()
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2d(
                    out_planes // 2,
                    out_planes // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=out_planes // 2,
                    bias=False),
                nn.BatchNorm2d(out_planes // 2), )
            self.skip = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    in_planes,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=in_planes,
                    bias=False),
                nn.BatchNorm2d(in_planes),
                nn.Conv2d(in_planes,
                          out_planes,
                          kernel_size=1,
                          bias=False),
                nn.BatchNorm2d(out_planes), )
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(
                    ConvBNRelu(in_planes, out_planes // 2, kernel=1))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(
                    ConvBNRelu(out_planes // 2, out_planes // 2, stride=stride))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(
                    ConvBNRelu(out_planes // 2, out_planes // 4, stride=stride))
            elif idx < block_num - 1:
                self.conv_list.append(
                    ConvBNRelu(out_planes // int(math.pow(2, idx)), out_planes
                               // int(math.pow(2, idx + 1))))
            else:
                self.conv_list.append(
                    ConvBNRelu(out_planes // int(math.pow(2, idx)), out_planes
                               // int(math.pow(2, idx))))

    def forward(self, x):
        out_list = []
        out = x
        for idx, conv in enumerate(self.conv_list):
            if idx == 0 and self.stride == 2:
                out = self.avd_layer(conv(out))
            else:
                out = conv(out)
            out_list.append(out)
        if self.stride == 2:
            x = self.skip(x)
        return torch.cat(out_list, dim=1) + x

class STDCNet(nn.Module):
    def __init__(self,
                 base=64,
                 layers=[4, 5, 3],
                 block_num=4,
                 type="cat",
                 pretrained="None"):
        super(STDCNet, self).__init__()
        if type == "cat":
            block = CatBottleneck
        elif type == "add":
            block = AddBottleneck
        self.layers = layers
        self.feat_channels = [base // 2, base, base * 4, base * 8, base * 16]
        self.features = self._make_layers(base, layers, block_num, block)

        # self.pretrained = pretrained
        # self.init_weight()

    def forward(self, x):
        out_feats = []

        # print(len(self.features))

        x = self.features[0](x)
        out_feats.append(x)
        x = self.features[1](x)
        out_feats.append(x)

        idx = [[2, 2 + self.layers[0]],
               [2 + self.layers[0], 2 + sum(self.layers[0:2])],
               [2 + sum(self.layers[0:2]), 2 + sum(self.layers)]]
        for start_idx, end_idx in idx:
            for i in range(start_idx, end_idx):
                x = self.features[i](x)
            out_feats.append(x)
        return out_feats

    def _make_layers(self, base, layers, block_num, block):
        features = []
        features += [ConvBNRelu(3, base // 2, 3, 2)]
        features += [ConvBNRelu(base // 2, base, 3, 2)]

        for i, layer in enumerate(layers):
            for j in range(layer):
                if i == 0 and j == 0:
                    features.append(block(base, base * 4, block_num, 2))
                elif j == 0:
                    features.append(
                        block(base * int(math.pow(2, i + 1)), base * int(
                            math.pow(2, i + 2)), block_num, 2))
                else:
                    features.append(
                        block(base * int(math.pow(2, i + 2)), base * int(
                            math.pow(2, i + 2)), block_num, 1))

        return nn.Sequential(*features)

if __name__ == "__main__":
    x = torch.randn(1, 3, 224, 224)
    # STDCNet813
    STDC1 = STDCNet(base=64, layers=[2, 2, 2])
    y1 = STDC1(x)
    print(len(y1))
    # STDCNet1446
    STDC2 = STDCNet(base=64, layers=[4, 5, 3])
    y2 = STDC2(x)
    print(len(y2))