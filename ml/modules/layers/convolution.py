from torch import nn


class Convolution(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride, padding, bias, conv_config):
        super(Convolution, self).__init__()

        if conv_config.name == 'conv':
            self.add_module('convolution', nn.Conv2d(in_channels, out_channels, conv_config.params.kernel_size,
                                                     stride, padding, bias=bias))
        elif conv_config.name == 'erf_conv':
            raise NotImplementedError()
        elif conv_config.name == 'inception_conv':
            raise NotImplementedError()
        elif conv_config.name == 'separable_conv':
            raise NotImplementedError()
        elif conv_config.name == 'separable_erf_conv':
            raise NotImplementedError()
        elif conv_config.name == 'inception_erf_conv':
            raise NotImplementedError()
        else:
            raise ValueError(f'Wrong conv name: {conv_config.name}')

# class ERFConv(nn.Sequential):
#     def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias, activation='relu'):
#         super().__init__()
#
#         self.activation = activation
#
#         self.conv3x1_1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1, 0), bias=True)
#         # init_weights(self.conv3x1_1, init_w='kaiming')
#
#         self.conv1x3_1 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1), bias=True)
#         # init_weights(self.conv1x3_1, init_w='kaiming')
#
#         self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)
#         # init_weights(self.bn1, init_w='kaiming')
#
#         self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1 * dilated, 0), bias=True,
#                                    dilation=(dilated, 1))
#         # init_weights(self.conv3x1_2, init_w='kaiming')
#
#         self.conv1x3_2 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1 * dilated), bias=True,
#                                    dilation=(1, dilated))
#         # init_weights(self.conv1x3_2, init_w='kaiming')
#
#         self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)
#         # init_weights(self.bn2, init_w='kaiming')
#
#         self.dropout = nn.Dropout2d(dropprob)
#
#     def forward(self, input):
#         output = self.conv3x1_1(input)
#         output = activation_fn(output, self.activation)
#         output = self.conv1x3_1(output)
#         output = self.bn1(output)
#         output = activation_fn(output, self.activation)
#
#         output = self.conv3x1_2(output)
#         output = activation_fn(output, self.activation)
#         output = self.conv1x3_2(output)
#         output = self.bn2(output)
#
#         if (self.dropout.p != 0):
#             output = self.dropout(output)
#
#         return activation_fn(output + input, self.activation)