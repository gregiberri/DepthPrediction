import torch.nn


class Criteria(torch.nn.Module):
    def __init__(self, criteria_config):
        super(Criteria, self).__init__()
        if criteria_config.name == 'mae' or criteria_config.name == 'l1':
            self.criteria = (torch.nn.L1Loss())
        elif criteria_config.name == 'mse' or criteria_config.name == 'l2':
            self.criteria = (torch.nn.MSELoss())
        elif criteria_config.name == 'crossentropy':
            self.criteria = (torch.nn.CrossEntropyLoss())
        elif criteria_config.name == 'smoothl1':
            self.criteria = (torch.nn.SmoothL1Loss())
        elif criteria_config.name == 'dorn':
            raise NotImplementedError()  # self.add_module('dorn_loss', DornLoss(criteria_config.params))
        else:
            raise ValueError(f'Wrong criteria name: {criteria_config.name}')

    def forward(self, pred, target):
        gt_mask = target > 0
        pred = pred[gt_mask]
        target = target[gt_mask]
        return self.criteria(pred, target)
