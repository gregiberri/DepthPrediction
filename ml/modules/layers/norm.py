from torch import nn


class Norm(nn.Sequential):
    def __init__(self, out_channels, norm_config):
        super(Norm, self).__init__()
        if norm_config.name == 'batch_norm':
            self.add_module('batch_norm', nn.BatchNorm2d(out_channels))
        elif norm_config.name == 'instance_norm':
            self.add_module('instance_norm', nn.InstanceNorm2d(out_channels))
        elif norm_config.name == 'group_norm':
            self.add_module('group_norm', nn.GroupNorm(out_channels / 2, out_channels))
        elif norm_config.name == 'no':
            pass
        else:
            raise ValueError(f'Wrong norm name: {norm_config.name}')
