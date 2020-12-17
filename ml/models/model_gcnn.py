import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.ERFNet import non_bottleneck_1d, UpsamplerBlock, DownsamplerBlock
from modules.guided_convolution import GuidedConvolution
from random import random


class GuidedCNN(nn.Module):
    def __init__(self, args):
        super(GuidedCNN, self).__init__()

        self.modality = args.input
        self.fusion_operation = args.gcnn_fusion_operation
        self.gcnn_train_gc_from_epoch = args.gcnn_train_gc_from_epoch
        self.args = args

        num_channels_superblock1 = args.gcnn_num_channels_sb1
        num_channels_superblock2 = args.gcnn_num_channels_sb2

        # GuideNet

        if 'rgb' in self.modality:
            channels_guidenet = args.first_channel_number[self.modality.index('r')]
            channels_guidenet += args.first_channel_number[self.modality.index('g')]
            channels_guidenet += args.first_channel_number[self.modality.index('b')]
            self.downsampler_img = DownsamplerBlock(3, channels_guidenet)
        elif 'g' in self.modality:
            channels_guidenet = args.first_channel_number[self.modality.index('g')]
            self.downsampler_img = DownsamplerBlock(1, channels_guidenet)

        if num_channels_superblock1 < channels_guidenet:
            raise Exception("The number of channels after initial downsampling can not be more than the number of "
                            "channels in the subsequent residual blocks.")

        self.guidenet_downsampler1 = DownsamplerBlock(channels_guidenet, num_channels_superblock1)
        self.guidenet_nonbt1dblock1a = non_bottleneck_1d(num_channels_superblock1, 0.03, 1)
        self.guidenet_nonbt1dblock1b = non_bottleneck_1d(num_channels_superblock1, 0.03, 1)
        self.guidenet_nonbt1dblock1c = non_bottleneck_1d(num_channels_superblock1, 0.03, 1)
        self.guidenet_nonbt1dblock1d = non_bottleneck_1d(num_channels_superblock1, 0.03, 1)
        self.guidenet_nonbt1dblock1e = non_bottleneck_1d(num_channels_superblock1, 0.03, 1)

        if num_channels_superblock2 < num_channels_superblock1:
            raise Exception("The number of channels in the first set of residual blocks can not be more than the "
                            "number of channels in the second set of residual blocks.")

        self.guidenet_downsampler2 = DownsamplerBlock(num_channels_superblock1, num_channels_superblock2)
        self.guidenet_nonbt1dblock2a = non_bottleneck_1d(num_channels_superblock2, 0.3, 1)
        self.guidenet_nonbt1dblock2b = non_bottleneck_1d(num_channels_superblock2, 0.3, 1)
        self.guidenet_nonbt1dblock2c = non_bottleneck_1d(num_channels_superblock2, 0.3, 1)
        self.guidenet_nonbt1dblock2d = non_bottleneck_1d(num_channels_superblock2, 0.3, 1)
        self.guidenet_nonbt1dblock2e = non_bottleneck_1d(num_channels_superblock2, 0.3, 1)
        self.guidenet_nonbt1dblock2f = non_bottleneck_1d(num_channels_superblock2, 0.3, 1)
        self.guidenet_nonbt1dblock2g = non_bottleneck_1d(num_channels_superblock2, 0.3, 1)
        self.guidenet_nonbt1dblock2h = non_bottleneck_1d(num_channels_superblock2, 0.3, 1)

        self.guidenet_deconv1 = UpsamplerBlock(num_channels_superblock2, num_channels_superblock1)
        self.guidenet_nonbt1dblock3a = non_bottleneck_1d(num_channels_superblock1, 0, 1)
        self.guidenet_nonbt1dblock3b = non_bottleneck_1d(num_channels_superblock1, 0, 1)

        self.guidenet_deconv2 = UpsamplerBlock(num_channels_superblock1, channels_guidenet)
        self.guidenet_nonbt1dblock4a = non_bottleneck_1d(channels_guidenet, 0, 1)
        self.guidenet_nonbt1dblock4b = non_bottleneck_1d(channels_guidenet, 0, 1)

        # DepthNet

        channels_depthnet = 0  # the number of channels after concatenating LIDAR depth and LIDAR intensity
        if 'd' in self.modality:
            channels = args.first_channel_number[self.modality.index('d')]
            self.downsampler_d = DownsamplerBlock(1, channels)
            channels_depthnet += channels
        if 'i' in self.modality:
            channels = args.first_channel_number[self.modality.index('i')]
            self.downsampler_i = DownsamplerBlock(1, channels)
            channels_depthnet += channels

        if self.fusion_operation == 'gc':
            self.depthnet_gc1 = GuidedConvolution(channels_depthnet, channels_depthnet, channels_guidenet, norm=args.gcnn_norm)
            num_input_channels_downsampler1 = channels_depthnet
        elif self.fusion_operation == 'c':
            num_input_channels_downsampler1 = channels_depthnet + channels_guidenet
        elif self.fusion_operation == 'a':
            if channels_guidenet != channels_depthnet:
                raise Exception("When addition is used as fusion operator, the number of channels in guidenet (sum of "
                                "R, G, B channels or number of grayscale channels) and depthnet (sum of the intensity "
                                "and depth channels) must be the same. Currently the number of depthnet channels is "
                                "{}, but the number of guidenet channels is {}.".format(channels_depthnet,
                                                                                        channels_guidenet))
            num_input_channels_downsampler1 = channels_guidenet

        self.depthnet_downsampler1 = DownsamplerBlock(num_input_channels_downsampler1, num_channels_superblock1)
        self.depthnet_nonbt1dblock1a = non_bottleneck_1d(num_channels_superblock1, 0.03, 1)
        self.depthnet_nonbt1dblock1b = non_bottleneck_1d(num_channels_superblock1, 0.03, 1)
        self.depthnet_nonbt1dblock1c = non_bottleneck_1d(num_channels_superblock1, 0.03, 1)
        self.depthnet_nonbt1dblock1d = non_bottleneck_1d(num_channels_superblock1, 0.03, 1)
        self.depthnet_nonbt1dblock1e = non_bottleneck_1d(num_channels_superblock1, 0.03, 1)

        if self.fusion_operation == 'gc':
            self.depthnet_gc2 = GuidedConvolution(in_channels=num_channels_superblock1,
                                                  out_channels=num_channels_superblock1,
                                                  guidance_channels=num_channels_superblock1, norm=args.gcnn_norm)
            num_input_channels_downsampler2 = num_channels_superblock1
        elif self.fusion_operation == 'c':
            num_input_channels_downsampler2 = num_channels_superblock1+num_channels_superblock1
        elif self.fusion_operation == 'a':
            num_input_channels_downsampler2 = num_channels_superblock1

        self.depthnet_downsampler2 = DownsamplerBlock(num_input_channels_downsampler2, num_channels_superblock2)
        self.depthnet_nonbt1dblock2a = non_bottleneck_1d(num_channels_superblock2, 0.3, 1)
        self.depthnet_nonbt1dblock2b = non_bottleneck_1d(num_channels_superblock2, 0.3, 1)
        self.depthnet_nonbt1dblock2c = non_bottleneck_1d(num_channels_superblock2, 0.3, 1)
        self.depthnet_nonbt1dblock2d = non_bottleneck_1d(num_channels_superblock2, 0.3, 1)
        self.depthnet_nonbt1dblock2e = non_bottleneck_1d(num_channels_superblock2, 0.3, 1)
        self.depthnet_nonbt1dblock2f = non_bottleneck_1d(num_channels_superblock2, 0.3, 1)
        self.depthnet_nonbt1dblock2g = non_bottleneck_1d(num_channels_superblock2, 0.3, 1)
        self.depthnet_nonbt1dblock2h = non_bottleneck_1d(num_channels_superblock2, 0.3, 1)

        self.depthnet_deconv1 = UpsamplerBlock(num_channels_superblock2, num_channels_superblock1)
        self.depthnet_nonbt1dblock3a = non_bottleneck_1d(num_channels_superblock1, 0, 1)
        self.depthnet_nonbt1dblock3b = non_bottleneck_1d(num_channels_superblock1, 0, 1)

        self.depthnet_deconv2 = UpsamplerBlock(num_channels_superblock1, channels_depthnet)
        self.depthnet_nonbt1dblock4a = non_bottleneck_1d(channels_depthnet, 0, 1)
        self.depthnet_nonbt1dblock4b = non_bottleneck_1d(channels_depthnet, 0, 1)

        self.depthnet_deconv3 = UpsamplerBlock(channels_depthnet, 16)
        self.conv = nn.Conv2d(16, 1, (3, 3), padding=1)

    def forward(self, x):
        # GuideNet
        if 'rgb' in self.modality:
            x1 = self.downsampler_img(x['rgb'])
        elif 'g' in self.modality:
            x1 = self.downsampler_img(x['g'])

        out_encoder1 = x1

        x1 = self.guidenet_downsampler1(x1)
        x1 = self.guidenet_nonbt1dblock1a(x1)
        x1 = self.guidenet_nonbt1dblock1b(x1)
        x1 = self.guidenet_nonbt1dblock1c(x1)
        x1 = self.guidenet_nonbt1dblock1d(x1)
        x1 = self.guidenet_nonbt1dblock1e(x1)

        out_encoder2 = x1

        x1 = self.guidenet_downsampler2(x1)
        x1 = self.guidenet_nonbt1dblock2a(x1)
        x1 = self.guidenet_nonbt1dblock2b(x1)
        x1 = self.guidenet_nonbt1dblock2c(x1)
        x1 = self.guidenet_nonbt1dblock2d(x1)
        x1 = self.guidenet_nonbt1dblock2e(x1)
        x1 = self.guidenet_nonbt1dblock2f(x1)
        x1 = self.guidenet_nonbt1dblock2g(x1)
        x1 = self.guidenet_nonbt1dblock2h(x1)

        out_guidenet3 = x1

        x1 = self.guidenet_deconv1(x1)
        x1 = self.guidenet_nonbt1dblock3a(x1)
        x1 = self.guidenet_nonbt1dblock3b(x1)

        x1 = x1 + out_encoder2
        out_guidenet2 = x1

        x1 = self.guidenet_deconv2(x1)
        x1 = self.guidenet_nonbt1dblock4a(x1)
        x1 = self.guidenet_nonbt1dblock4b(x1)

        out_guidenet1 = x1 + out_encoder1

        # DepthNet
        if 'd' in self.modality:
            downsampler_d = self.downsampler_d(x['sampledraw'])
        if 'i' in self.modality:
            downsampler_i = self.downsampler_i(x['lidar_intensity'])

        if ('d' in self.modality) and ('i' in self.modality):
            downsampler_depthnet = torch.cat((downsampler_d, downsampler_i), 1)
        elif 'd' in self.modality:
            downsampler_depthnet = downsampler_d
        elif 'i' in self.modality:
            downsampler_depthnet = downsampler_i
        else:
            raise Exception("Modality neither contains \'d\', nor \'i\'.")

        # print(downsampler_depthnet.size())

        use_addition_instead_of_gc = self.args.current_epoch < self.gcnn_train_gc_from_epoch

        if random() < 0.01:
            print("Use addition instead of GC: {} {} {}".format(use_addition_instead_of_gc, self.args.current_epoch, self.gcnn_train_gc_from_epoch))

        if self.fusion_operation == 'gc':
            if use_addition_instead_of_gc:
                x2 = downsampler_depthnet + out_guidenet1
            else:
                x2 = self.depthnet_gc1({'data': downsampler_depthnet, 'guidance': out_guidenet1})
        elif self.fusion_operation == 'a':
            x2 = downsampler_depthnet + out_guidenet1
        elif self.fusion_operation == 'c':
            x2 = torch.cat((downsampler_depthnet, out_guidenet1), 1)

        x2 = self.depthnet_downsampler1(x2)

        # print(x2.size())

        x2 = self.depthnet_nonbt1dblock1a(x2)
        x2 = self.depthnet_nonbt1dblock1b(x2)
        x2 = self.depthnet_nonbt1dblock1c(x2)
        x2 = self.depthnet_nonbt1dblock1d(x2)
        x2 = self.depthnet_nonbt1dblock1e(x2)

        residual2 = x2

        if self.fusion_operation == 'gc':
            if use_addition_instead_of_gc:
                x2 = x2 + out_guidenet2
            else:
                x2 = self.depthnet_gc2({'data': x2, 'guidance': out_guidenet2})
        elif self.fusion_operation == 'a':
            x2 = x2 + out_guidenet2
        elif self.fusion_operation == 'c':
            x2 = torch.cat((x2, out_guidenet2), 1)

        x2 = self.depthnet_downsampler2(x2)

        # print(x2.size())

        x2 = self.depthnet_nonbt1dblock2a(x2)
        x2 = self.depthnet_nonbt1dblock2b(x2)
        x2 = self.depthnet_nonbt1dblock2c(x2)
        x2 = self.depthnet_nonbt1dblock2d(x2)
        x2 = self.depthnet_nonbt1dblock2e(x2)
        x2 = self.depthnet_nonbt1dblock2f(x2)
        x2 = self.depthnet_nonbt1dblock2g(x2)
        x2 = self.depthnet_nonbt1dblock2h(x2)

        x2 = x2 + out_guidenet3

        x2 = self.depthnet_deconv1(x2)

        # print(x2.size())

        x2 = self.depthnet_nonbt1dblock3a(x2)
        x2 = self.depthnet_nonbt1dblock3b(x2)

        x2 = x2 + residual2

        x2 = self.depthnet_deconv2(x2)

        # print(x2.size())

        x2 = self.depthnet_nonbt1dblock4a(x2)
        x2 = self.depthnet_nonbt1dblock4b(x2)

        x2 = x2 + downsampler_depthnet

        x2 = self.depthnet_deconv3(x2)

        # print(x2.size())

        y = self.conv(x2)

        # print(y.size())

        if self.training:
            return 100 * y, None, None, None, None, None, None
        else:
            min_distance = 0.9
            return F.relu(100 * y - min_distance) + min_distance, None, None, \
                   None, None, None, None  # the minimum range of Velodyne is around 3 feet ~= 0.9m

