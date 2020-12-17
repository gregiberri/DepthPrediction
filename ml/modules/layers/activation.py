from torch import nn


class Activation(nn.Sequential):
    def __init__(self, activation_config):
        super(Activation, self).__init__()
        if activation_config.name == 'relu':
            self.add_module('relu', nn.ReLU())
        elif activation_config.name == 'leaky_relu':
            self.add_module('leaky_relu', nn.LeakyReLU(negative_slope=activation_config.params.negative_slope))
        elif activation_config.name == 'no':
            pass
        else:
            raise ValueError(f'Wrong activation name: {activation_config.name}')
