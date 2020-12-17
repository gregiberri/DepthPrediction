import torch.nn as nn
from modules.ERFNet import init_weights


class ResidualBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1, relu: float = 0.2,
                 norm: str = 'batchnorm', init_w: str = 'normal', stride: int = 1):
        """ This block is an implementation of Fig. 2 with convolutional layers from
            He et al.: "Deep Residual Learning for Image Recognition"

            Parameters
            ----------
                in_channels : int
                    the number of channels of the input images
                out_channels : int
                    the number of output channels of guided convolution
                kernel_size : int
                    a squared kernel of shape kernel_size x kernel_size is used to convolve the input images,
                    default value: 3
                padding : int
                    padding used when calculating the guided convolution, default value: 1
                relu : float
                    if relu > 0, the activation function is leaky ReLU with a parameter specified by this argument,
                    otherwise vanilla ReLU is used
                norm : str
                    type of normalisation, either 'batchnorm' (default), "groupnorm' or 'relu'
                init_w : str
                    weight initialisation strategy of ERFNet.init_weights, either 'normal' (default) or 'kaiming'

            Raises
            ------
                Exception
                    if the value of relu or norm is different from the ones described above
            """

        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride)
        self.relu1 = create_relu(relu)
        self.norm1 = create_norm(norm, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.relu2 = create_relu(relu)
        self.norm2 = create_norm(norm, out_channels)

        if (in_channels != out_channels) or (stride > 1):
            self.shortcut_convolution = nn.Conv2d(in_channels, out_channels, 1, stride=stride)
        else:
            self.shortcut_convolution = None

        init_weights(self.conv1, init_w)
        init_weights(self.relu1, init_w)
        init_weights(self.norm1, init_w)
        init_weights(self.conv2, init_w)
        init_weights(self.relu2, init_w)
        init_weights(self.norm2, init_w)
        if self.shortcut_convolution is not None:
            init_weights(self.shortcut_convolution, init_w)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.relu1(x1)
        if self.norm1 is not None:
            x1 = self.norm1(x1)

        x1 = self.conv2(x1)

        if self.shortcut_convolution is not None:
            x2 = self.shortcut_convolution(x)
        else:
            x2 = x

        x3 = self.relu2(x1 + x2)
        if self.norm2 is not None:
            x3 = self.norm2(x3)

        return x3


def create_relu(relu: float):
    if relu > 0:
        return nn.LeakyReLU(relu, inplace=True)
    else:
        return nn.ReLU(inplace=True)


def create_norm(norm: str, num_channels: int):
    if norm == 'batchnorm':
        return nn.BatchNorm2d(num_channels)
    elif norm == 'groupnorm':
        return nn.GroupNorm(int(num_channels / 2), num_channels)
    elif norm == 'none':
        return None
    else:
        raise Exception('invalid value for the type of normalisation (norm) when initializing GuidedConvolution: '
                        '{}'.format(norm))
