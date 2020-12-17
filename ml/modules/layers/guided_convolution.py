import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.ERFNet import init_weights


class GuidedConvolution(nn.Module):


    def __init__(self, in_channels: int, out_channels: int, guidance_channels, kernel_size=3, padding=1,
                 kgl_kernel_size=3, activation='leaky_relu', norm='batchnorm', init_w='normal', leakage=0.2):
        """ Guided convolution based on the description in
            Jie Tang, Fei-Peng Tian, Wei Feng, Jian Li, and Ping Tan: "Learning Guided Convolutional Network for Depth
            Completion", arXiv:1908.01238v1 [cs.CV], 3 Aug 2019

            Parameters
            ----------
                in_channels : int
                    the number of channels of the input images
                out_channels : int
                    the number of output channels of guided convolution
                guidance_channels : int
                    the number of input channels of the kernel-generation layer
                kernel_size : int
                    a squared kernel of shape kernel_size x kernel_size is used to convolve the input images, note that
                    this is a guided kernel, i.e., it is not constant for all the spatial locations, default value: 3
                padding : int
                    padding used when calculating the guided convolution, default value: 1
                kgl_kernel_size : int
                    the size of kernel used in the kernel-generation layer (KGL) that produces the channel-wise kernel
                    (in this case, KGL is a convolutional layer), it must be an odd number
                activation : str
                    the subtype of the ReLU activation function, either 'leaky' (default) or 'vanilla'
                norm : str
                    type of normalisation, either 'batchnorm' (default), "groupnorm' or 'relu'
                init_w : str
                    weight initialisation strategy of ERFNet.init_weights, either 'normal' (default) or 'kaiming'

            Raises
            ------
                Exception
                    if kgl_kernel_size is an even number or the value of relu or norm is different from the ones
                    described above
            """

        super(GuidedConvolution, self).__init__()

        self.guided_conv_kernel_width = kernel_size
        self.guided_conv_kernel_height = kernel_size
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.padding = padding

        if kgl_kernel_size % 2 != 1:
            raise Exception('kgl_kernel_size in GuidedConvolution should be an odd integer')

        self.kgl_channelwise = nn.Conv2d(guidance_channels, in_channels, kgl_kernel_size,
                                         padding=padding + int(kgl_kernel_size / 2))
        self.kgl_crosschannel = nn.Linear(guidance_channels, in_channels * out_channels)

        if activation == 'leaky_relu':
            self.relu = nn.LeakyReLU(leakage, inplace=True)
        elif activation == 'relu':
            self.relu = nn.ReLU(inplace=True)
        else:
            raise Exception('invalid value for the type of ReLU when initializing GuidedConvolution: {}'.format(activation))

        if norm == 'batchnorm':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'groupnorm':
            self.norm = nn.GroupNorm(int(out_channels / 2), out_channels)
        elif norm == 'none':
            self.norm = None
        else:
            raise Exception('invalid value for the type of normalisation (norm) when initializing GuidedConvolution: '
                            '{}'.format(norm))

        init_weights(self.kgl_channelwise, init_w)
        init_weights(self.kgl_crosschannel, init_w)
        init_weights(self.relu, init_w)
        if self.norm is not None:
            init_weights(self.norm, init_w)

    def forward(self, x):
        kernel = self.kgl_channelwise(x['guidance'])  # Generate spatially variant channel-wise kernel
        x1 = self._calculate_channel_wise_convolution(x['data'], kernel)

        kernel_crosschannel = self.kgl_crosschannel(self._mean_intensity(x['guidance']))  # Generate cross-channel kernel
        kernel_crosschannel = kernel_crosschannel.view(-1, self.out_channels, self.in_channels, 1, 1)
        x2 = self._calculate_cross_channel_convolution(x1, kernel_crosschannel)

        x3 = self.relu(x2)

        if self.norm is None:
            return x3
        else:
            return self.norm(x3)

    def _calculate_channel_wise_convolution(self, x, kernel):
        batch_size, num_channels, image_size_y, image_size_x = x.size()

        padding_rows = torch.zeros((self.padding, image_size_x), device=torch.device("cuda"))
        padding_cols = torch.zeros((image_size_y + 2 * self.padding, self.padding), device=torch.device("cuda"))

        x_convolved = torch.zeros((batch_size, num_channels,
                                   image_size_y - self.guided_conv_kernel_height + 1 + 2 * self.padding,
                                   image_size_x - self.guided_conv_kernel_width + 1 + 2 * self.padding),
                                  device=torch.device("cuda"))
        for idx in range(batch_size):
            for channel in range(num_channels):
                x_padded = torch.cat((padding_rows, x[idx, channel, :, :], padding_rows))
                x_padded = torch.cat((padding_cols, x_padded, padding_cols), 1)
                x_mul_kernel = x_padded * kernel[idx, channel, :, :]

                ones = torch.ones((1, 1, self.guided_conv_kernel_height, self.guided_conv_kernel_width),
                                  device=torch.device("cuda"))

                x_convolved[idx, channel, :, :] = F.conv2d(x_mul_kernel.view([1, 1] + list(x_mul_kernel.size())), ones)

        return x_convolved

    def _mean_intensity(self, x):
        ''' calculates average intensity per frame and channel
        '''
        batch_size, num_channels, _, _ = x.size()

        mean_intensity = torch.zeros((batch_size, num_channels), device=torch.device("cuda"))
        for idx in range(batch_size):
            for channel in range(num_channels):
                mean_intensity[idx, channel] = torch.mean(x[idx, channel, :, :])

        return mean_intensity

    def _calculate_cross_channel_convolution(self, x, kernel):

        batch_size, num_channels, image_size_y, image_size_x = x.size()

        x_convolved = torch.zeros((batch_size, self.out_channels, image_size_y, image_size_x),
                                  device=torch.device("cuda"))
        for idx in range(batch_size):
            x_convolved[idx, :, :, :] = F.conv2d(x[idx].view([1] + list(x[idx].size())), kernel[idx])[0, :, :, :]

        return x_convolved
