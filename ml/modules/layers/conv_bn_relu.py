from torch import nn

from ml.modules.layers.activation import Activation
from ml.modules.layers.convolution import Convolution
from ml.modules.layers.norm import Norm
from ml.modules.layers.utils import init_weights


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride=1, padding=0, conv_norm_act_config=None):
        super(ConvNormAct, self).__init__()

        norm = conv_norm_act_config.norm
        init_w = conv_norm_act_config.init_w or 'normal'

        bias = not norm

        # add the convolution
        self.add_module('conv', Convolution(in_channels, out_channels, stride, padding, bias,
                                            conv_config=conv_norm_act_config.conv))

        # add normalization
        self.add_module('norm', Norm(out_channels, norm_config=conv_norm_act_config.norm))

        # add activation
        self.add_module('act', Activation(activation_config=conv_norm_act_config.activation))

        # initialize the weights
        for m in self.modules():
            init_weights(m, init_w)


def conv_bn_relu(in_channels, out_channels, kernel_size, \
                 stride=1, padding=0, norm='batchnorm', activation='relu', init_w='normal',
                 use_erf_conv=False, conv_type='conv', **kwargs):
    bias = not norm
    layers = []
    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                            padding, bias=bias))
    if norm == 'batchnorm':
        layers.append(nn.BatchNorm2d(out_channels))
    elif norm == 'groupnorm':
        layers.append(nn.GroupNorm(out_channels / 2, out_channels))
    if activation == 'leaky_relu':
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    elif activation == 'relu':
        layers.append(nn.ReLU(inplace=True))
    layers = nn.Sequential(*layers)

    # initialize the weights
    for m in layers.modules():
        init_weights(m, init_w)

    return layers


def convt_bn_relu(in_channels, out_channels, kernel_size, \
                  stride=1, padding=0, output_padding=0, norm='batchnorm', activation='relu', init_w='normal',
                  **kwargs):
    bias = not norm
    layers = []
    layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                     stride, padding, output_padding, bias=bias))
    if norm == 'batchnorm':
        layers.append(nn.BatchNorm2d(out_channels))
    elif norm == 'groupnorm':
        layers.append(nn.GroupNorm(out_channels / 2, out_channels))
    if activation == 'leaky_relu':
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    elif activation == 'relu':
        layers.append(nn.ReLU(inplace=True))
    layers = nn.Sequential(*layers)

    # initialize the weights
    for m in layers.modules():
        init_weights(m, init_w)

    return layers
