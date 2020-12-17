import torch

from ml.modules.heads.dorn import OrdinalRegression
from ml.modules.heads.regression import Regression
from ml.modules.layers.bifpn import BiFpn


class Top(torch.nn.Sequential):
    def __init__(self, top_config):
        super(Top, self).__init__()
        if top_config.name == 'bifpn_last':
            self.add_module('regression', BiFpn(top_config.params))

        elif top_config.name == 'bifpn_fusion':
            raise NotImplementedError()
        else:
            raise ValueError(f'Wrong head name: {top_config.name}')
