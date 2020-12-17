from types import SimpleNamespace

import torch
from torch import nn

from modules.bifpn import BiFpn, Decoder
from modules.conv_bn_relu import conv_bn_relu
from modules.erfdet_modules import ERFEncoder


class Det(nn.Module):
    def __init__(self, args):
        super(Det, self).__init__()

        self.modality = args.input
        self.args = args
        channels = 0
        if 'd' in self.modality:
            channels += 1
        if 'i' in self.modality:
            channels += 1
        if 'rgb' in self.modality:
            channels += 3
        elif 'g' in self.modality:
            channels += 1

        # backbone
        self.depth_net = ERFEncoder(in_channels=channels, activation=args.activation)

        # input encode
        # local_channels_in = 2 if self.modality not in ["rgb", "g"] else 1
        self.conv_bn_relu = conv_bn_relu(in_channels=channels, out_channels=16, kernel_size=3, stride=1,
                                         padding=1, norm='no', activation=args.activation, init_w='kaiming')

        # BiFPN-s
        bifpn_feature_numbers = [16, 32, 64, 128]
        bifpn_number = args.bifpn_number
        self.bifpn = BiFpn(bifpn_feature_numbers, cell_number=bifpn_number, backbone_depth=4)

        self.decoder = Decoder(bifpn_feature_numbers, backbone_depth=4)

    def forward(self, x):
        # input definition
        if 'd' in self.modality:
            d = x['sampledraw']
        if 'i' in self.modality:
            i = x['lidar_intensity']
        if 'rgb' in self.modality:
            img = x['rgb']
        elif 'g' in self.modality:
            img = x['g']

        if self.modality == 'rgbd' or self.modality == 'gd':
            input_data = torch.cat((d, img), 1)
        elif self.modality == 'rgbi' or self.modality == 'gi':
            input_data = torch.cat((i, img), 1)
        elif self.modality == 'rgbdi' or self.modality == 'gdi':
            input_data = torch.cat((i, d, img), 1)
        else:
            input_data = d if (self.modality == 'd') else i if (self.modality == 'i') else img

        # 1. Backbone
        p1, p2, p3 = self.depth_net(input_data)

        # 2. Input encoding
        p0 = self.conv_bn_relu(input_data)

        # 3. BiFPN layers
        bifpn_features = self.bifpn([p0, p1, p2, p3])

        # 4. Decode to upsample the prediction
        out = self.decoder(bifpn_features)

        return out


if __name__ == '__main__':
    args = SimpleNamespace()
    args.input = 'gd'
    args.activation = 'relu'
    args.bifpn_number = 2

    det_model = Det(args)

    image = torch.rand([1, 1, 352, 1248], dtype=torch.float32)
    depth = torch.rand([1, 1, 352, 1248], dtype=torch.float32)

    pred = det_model({'g': image, 'sampledraw': depth})

    asd = 1
