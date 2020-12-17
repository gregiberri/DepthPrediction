# ERFNet full model definition for Pytorch
# Sept 2017
# Eduardo Romera
#######################

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init




# def activation_fn(inputs, activation):
#     if activation == 'relu':
#         out = F.relu(inputs)
#     elif activation == 'leaky_relu':
#         out = F.leaky_relu(inputs, 0.2)
#     else:
#         out = inputs
#
#     return out
from ml.modules.layers.activation import Activation


class DownsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput, activation_config):
        super().__init__()

        if noutput > ninput:
            self.conv = nn.Conv2d(ninput, noutput - ninput, (3, 3), stride=2, padding=1, bias=True)
        else:
            self.conv = None

        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

        self.activation = Activation(activation_config)

    def forward(self, input):

        if self.conv is not None:
            output = torch.cat([self.conv(input), self.pool(input)], 1)
        else:
            output = self.pool(input)

        output = self.bn(output)

        output = self.activation(output)
        return output


class non_bottleneck_1d(nn.Module):
    def __init__(self, chann, dropprob, dilated, activation_config):
        super().__init__()

        self.activation = activation

        self.conv3x1_1 = torch.nn.Sequential(
            nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1, 0), bias=True),
            Activation(activation_config))


        self.conv1x3_1 = torch.nn.nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1), bias=True)
        # init_weights(self.conv1x3_1, init_w='kaiming')

        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)
        # init_weights(self.bn1, init_w='kaiming')

        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1 * dilated, 0), bias=True,
                                   dilation=(dilated, 1))
        # init_weights(self.conv3x1_2, init_w='kaiming')

        self.conv1x3_2 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1 * dilated), bias=True,
                                   dilation=(1, dilated))
        # init_weights(self.conv1x3_2, init_w='kaiming')

        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)
        # init_weights(self.bn2, init_w='kaiming')

        self.dropout = nn.Dropout2d(dropprob)



    def forward(self, input):
        output = self.conv3x1_1(input)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = activation_fn(output, self.activation)

        output = self.conv3x1_2(output)
        output = activation_fn(output, self.activation)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if (self.dropout.p != 0):
            output = self.dropout(output)

        return activation_fn(output + input, self.activation)


class ERFEncoder(nn.Module):
    def __init__(self, in_channels, activation='relu'):
        super().__init__()
        chans = 32

        self.initial_block = DownsamplerBlock(in_channels, chans, activation=activation)

        self.p1_module = nn.ModuleList()
        self.p1_module.append(DownsamplerBlock(chans, 64, activation=activation))
        for x in range(0, 5):
            self.p1_module.append(non_bottleneck_1d(64, 0.03, 1, activation=activation))

        self.p2_module = nn.ModuleList()
        self.p2_module.append(DownsamplerBlock(64, 128, activation=activation))
        for x in range(0, 2):
            self.p2_module.append(non_bottleneck_1d(128, 0.3, 2, activation))
            self.p2_module.append(non_bottleneck_1d(128, 0.3, 4, activation))
            self.p2_module.append(non_bottleneck_1d(128, 0.3, 8, activation))
            self.p2_module.append(non_bottleneck_1d(128, 0.3, 16, activation))

    def forward(self, input):
        p1 = output = self.initial_block(input)

        for layer in self.p1_module:
            output = layer(output)
        p2 = output

        for layer in self.p2_module:
            output = layer(output)
        p3 = output

        return p1, p2, p3


class UpsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput, activation='relu'):
        super().__init__()
        self.activation = activation

        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return activation_fn(output, self.activation)


class ERFDecoder(nn.Module):
    def __init__(self, num_classes, activation='relu'):
        super().__init__()

        self.actiation = activation

        self.layer1 = UpsamplerBlock(128, 64, activation)
        self.layer2 = non_bottleneck_1d(64, 0, 1, activation)
        self.layer3 = non_bottleneck_1d(64, 0, 1, activation)  # 64x64x304

        self.layer4 = UpsamplerBlock(64, 32, activation)
        self.layer5 = non_bottleneck_1d(32, 0, 1, activation)
        self.layer6 = non_bottleneck_1d(32, 0, 1, activation)  # 32x128x608

        self.output_conv = nn.ConvTranspose2d(32, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True)

    def forward(self, input):
        output = input
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        p2 = output
        output = self.layer4(output)
        output = self.layer5(output)
        output = self.layer6(output)
        p1 = output

        p0 = self.output_conv(output)

        return p0, p1, p2


# class Det(nn.Module):
#     def __init__(self, in_channels=1, out_channels=1, activation='relu'):  # use encoder to pass pretrained encoder
#         super().__init__()
#         self.encoder = Encoder(in_channels, out_channels, activation)
#
#     def forward(self, input):
#         p1, p2, p3 = self.encoder(input)
#         return p1, p2, p3
