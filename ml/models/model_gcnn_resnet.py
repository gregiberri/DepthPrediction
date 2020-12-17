import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.guided_convolution import GuidedConvolution
from modules.residual_block import create_relu, create_norm
from modules.residual_block import ResidualBlock
from models.DCNet import convt_bn_relu


class GuidedCNN_ResNet(nn.Module):
    def __init__(self, args):
        super(GuidedCNN_ResNet, self).__init__()

        self.modality = args.input
        channels_rblock_5 = args.gcnn_channels_rblock5

        # GuideNet

        if 'rgb' in self.modality:
            input_channels = 3
            channels_guidenet = args.first_channel_number[self.modality.index('r')]
            channels_guidenet += args.first_channel_number[self.modality.index('g')]
            channels_guidenet += args.first_channel_number[self.modality.index('b')]
        elif 'g' in self.modality:
            input_channels = 1
            channels_guidenet = args.first_channel_number[self.modality.index('g')]
        else:
            raise Exception("Inappropriate modality")

        self.guidenet_initial_conv = nn.Conv2d(input_channels, channels_guidenet, 3, padding=1)
        self.guidenet_initial_relu = create_relu(0.2)
        self.guidenet_initial_norm = create_norm('batchnorm', channels_guidenet)

        self.guidenet_rblock_1_1 = ResidualBlock(channels_guidenet, 64, stride=2)
        self.guidenet_rblock_1_2 = ResidualBlock(64, 64)

        self.guidenet_rblock_2_1 = ResidualBlock(64, 128, stride=2)
        self.guidenet_rblock_2_2 = ResidualBlock(128, 128)

        self.guidenet_rblock_3_1 = ResidualBlock(128, 256, stride=2)
        self.guidenet_rblock_3_2 = ResidualBlock(256, 256)

        self.guidenet_rblock_4_1 = ResidualBlock(256, 512, stride=2)
        self.guidenet_rblock_4_2 = ResidualBlock(512, 512)

        self.guidenet_rblock_5_1 = ResidualBlock(512, channels_rblock_5, stride=2)
        self.guidenet_rblock_5_2 = ResidualBlock(channels_rblock_5, channels_rblock_5)

        self.guidenet_deconv5 = convt_bn_relu(channels_rblock_5, 512, 3, stride=2, padding=1, output_padding=1)
        self.guidenet_deconv4 = convt_bn_relu(512, 256, 3, stride=2, padding=1, output_padding=1)
        self.guidenet_deconv3 = convt_bn_relu(256, 128, 3, stride=2, padding=1, output_padding=1)
        self.guidenet_deconv2 = convt_bn_relu(128, 64, 3, stride=2, padding=1, output_padding=1)

        # DepthNet

        channels_depthnet = 0  # the number of channels after concatenating LIDAR depth and LIDAR intensity
        if 'd' in self.modality:
            channels = args.first_channel_number[self.modality.index('d')]
            self.depthnet_initial_conv_d = nn.Conv2d(1, channels, 3, padding=1)
            self.depthnet_initial_relu_d = create_relu(0.2)
            self.depthnet_initial_norm_d = create_norm('batchnorm', channels)
            channels_depthnet += channels
        if 'i' in self.modality:
            channels = args.first_channel_number[self.modality.index('i')]
            self.depthnet_initial_conv_i = nn.Conv2d(1, channels, 3, padding=1)
            self.depthnet_initial_relu_i = create_relu(0.2)
            self.depthnet_initial_norm_i = create_norm('batchnorm', channels)
            channels_depthnet += channels

        self.depthnet_rblock_1_1 = ResidualBlock(channels_depthnet, 64, stride=2)
        self.depthnet_rblock_1_2 = ResidualBlock(64, 64)

        self.depthnet_gc1 = GuidedConvolution(64, 64, 64, norm=args.gcnn_norm)

        self.depthnet_rblock_2_1 = ResidualBlock(64, 128, stride=2)
        self.depthnet_rblock_2_2 = ResidualBlock(128, 128)

        self.depthnet_gc2 = GuidedConvolution(128, 128, 128, norm=args.gcnn_norm)

        self.depthnet_rblock_3_1 = ResidualBlock(128, 256, stride=2)
        self.depthnet_rblock_3_2 = ResidualBlock(256, 256)

        self.depthnet_gc3 = GuidedConvolution(256, 256, 256, norm=args.gcnn_norm)

        self.depthnet_rblock_4_1 = ResidualBlock(256, 512, stride=2)
        self.depthnet_rblock_4_2 = ResidualBlock(512, 512)

        self.depthnet_gc4 = GuidedConvolution(512, 512, 512, norm=args.gcnn_norm)

        self.depthnet_rblock_5_1 = ResidualBlock(512, channels_rblock_5, stride=2)
        self.depthnet_rblock_5_2 = ResidualBlock(channels_rblock_5, channels_rblock_5)

        self.depthnet_deconv5 = convt_bn_relu(channels_rblock_5, 512, 3, stride=2, padding=1, output_padding=1)
        self.depthnet_deconv4 = convt_bn_relu(512, 256, 3, stride=2, padding=1, output_padding=1)
        self.depthnet_deconv3 = convt_bn_relu(256, 128, 3, stride=2, padding=1, output_padding=1)
        self.depthnet_deconv2 = convt_bn_relu(128, 64, 3, stride=2, padding=1, output_padding=1)
        self.depthnet_deconv1 = convt_bn_relu(64, 64, 3, stride=2, padding=1, output_padding=1)

        self.depthnet_final_conv = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, x):
        # GuideNet
        if 'rgb' in self.modality:
            x1 = self.guidenet_initial_conv(x['rgb'])
        elif 'g' in self.modality:
            x1 = self.guidenet_initial_conv(x['g'])

        x1 = self.guidenet_initial_relu(x1)
        x1 = self.guidenet_initial_norm(x1)

        x1 = self.guidenet_rblock_1_1(x1)
        x1 = self.guidenet_rblock_1_2(x1)

        out_encoder1 = x1

        x1 = self.guidenet_rblock_2_1(x1)
        x1 = self.guidenet_rblock_2_2(x1)

        out_encoder2 = x1

        x1 = self.guidenet_rblock_3_1(x1)
        x1 = self.guidenet_rblock_3_2(x1)

        out_encoder3 = x1

        x1 = self.guidenet_rblock_4_1(x1)
        x1 = self.guidenet_rblock_4_2(x1)

        out_encoder4 = x1

        x1 = self.guidenet_rblock_5_1(x1)
        x1 = self.guidenet_rblock_5_2(x1)

        out_guidenet5 = x1

        x1 = self.guidenet_deconv5(x1)
        x1 = x1 + out_encoder4
        out_guidenet4 = x1

        x1 = self.guidenet_deconv4(x1)
        x1 = x1 + out_encoder3
        out_guidenet3 = x1

        x1 = self.guidenet_deconv3(x1)
        x1 = x1 + out_encoder2
        out_guidenet2 = x1

        x1 = self.guidenet_deconv2(x1)
        x1 = x1 + out_encoder1
        out_guidenet1 = x1

        # DepthNet
        if 'd' in self.modality:
            x_d = self.depthnet_initial_conv_d(x['sampledraw'])
            x_d = self.depthnet_initial_relu_d(x_d)
            x_d = self.depthnet_initial_norm_d(x_d)
        if 'i' in self.modality:
            x_i = self.depthnet_initial_conv_d(x['lidar_intensity'])
            x_i = self.depthnet_initial_relu_d(x_i)
            x_i = self.depthnet_initial_norm_d(x_i)

        if ('d' in self.modality) and ('i' in self.modality):
            x2 = torch.cat((x_d, x_i), 1)
        elif 'd' in self.modality:
            x2 = x_d
        elif 'i' in self.modality:
            x2 = x_i
        else:
            raise Exception("Modality neither contains \'d\', nor \'i\'.")

        x2 = self.depthnet_rblock_1_1(x2)
        x2 = self.depthnet_rblock_1_2(x2)
        out_depthnet1 = x2
        x2 = self.depthnet_gc1({'data': x2, 'guidance': out_guidenet1})

        x2 = self.depthnet_rblock_2_1(x2)
        x2 = self.depthnet_rblock_2_2(x2)
        out_depthnet2 = x2
        x2 = self.depthnet_gc2({'data': x2, 'guidance': out_guidenet2})

        x2 = self.depthnet_rblock_3_1(x2)
        x2 = self.depthnet_rblock_3_2(x2)
        out_depthnet3 = x2
        x2 = self.depthnet_gc3({'data': x2, 'guidance': out_guidenet3})

        x2 = self.depthnet_rblock_4_1(x2)
        x2 = self.depthnet_rblock_4_2(x2)
        out_depthnet4 = x2
        x2 = self.depthnet_gc4({'data': x2, 'guidance': out_guidenet4})

        x2 = self.depthnet_rblock_5_1(x2)
        x2 = self.depthnet_rblock_5_2(x2)

        x2 = x2 + out_guidenet5
        x2 = self.depthnet_deconv5(x2)

        x2 = x2+out_depthnet4
        x2 = self.depthnet_deconv4(x2)

        x2 = x2 + out_depthnet3
        x2 = self.depthnet_deconv3(x2)

        x2 = x2 + out_depthnet2
        x2 = self.depthnet_deconv2(x2)

        x2 = x2 + out_depthnet1
        x2 = self.depthnet_deconv1(x2)

        y = self.depthnet_final_conv(x2)

        if self.training:
            return 100 * y, None, None, None, None, None, None
        else:
            min_distance = 0.9
            return F.relu(100 * y - min_distance) + min_distance, None, None, \
                   None, None, None, None  # the minimum range of Velodyne is around 3 feet ~= 0.9m
