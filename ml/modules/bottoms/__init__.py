import torch
from torch.nn import ModuleDict

from ml.modules.layers.conv_bn_relu import ConvNormAct


class Bottom(torch.nn.Module):
    def __init__(self, bottom_config):
        super(Bottom, self).__init__()
        self.bottom_config_name = bottom_config.name

        if bottom_config.name == 'early_fusion':
            self.fusion = ConvNormAct(bottom_config.params.input_channels,
                                      bottom_config.params.bottom_features,
                                      stride=1,
                                      padding=1,
                                      conv_norm_act_config=bottom_config.params)
        elif bottom_config.name == 'late_fusion':
            raise NotImplementedError()
        elif bottom_config.name == 'early_fusion_sparsity_invariant':
            raise NotImplementedError()
        else:
            raise ValueError(f'Wrong bottom name: {bottom_config.name}')

    def forward(self, image, depth=None):
        if depth != None:
            if 'early' in self.bottom_config_name:
                input = torch.cat([image, depth], dim=1)
                out = self.fusion(input)
            elif 'late' in self.bottom_config_name:
                raise NotImplementedError('Depth and image bottom separately')
            else:
                raise ValueError('Bottom name should contain `early` or `late` if there is depth.')
        else:
            raise NotImplementedError('Only image bottom')

        return out
