import torch

from ml.modules.heads.dorn import OrdinalRegression
from ml.modules.heads.regression import Regression


class Head(torch.nn.Sequential):
    def __init__(self, head_config):
        super(Head, self).__init__()
        if head_config.name == 'regression':
            self.add_module('regression', Regression(head_config.params))
        elif head_config.name == 'dorn':
            self.add_module('dorn', OrdinalRegression())
        elif head_config.name == 'dorn_regression':
            raise NotImplementedError()
        else:
            raise ValueError(f'Wrong head name: {head_config.name}')
