import torch
import torch.nn

from ml.criterias import Criteria


class Regression(torch.nn.Module):
    def __init__(self, regression_config):
        super().__init__()
        backbone_features = regression_config.backbone_features
        input_size = regression_config.input_size
        regression_criteria = regression_config.regression_criteria
        scaling_mode = regression_config.scaling_method or 'bilinear'

        self.pred_layer = torch.nn.Conv2d(in_channels=backbone_features,
                                          out_channels=1,
                                          kernel_size=1)
        self.sample_layer = torch.nn.Upsample(input_size, mode=scaling_mode)
        self.reg_criterion = Criteria(regression_criteria)

    def forward(self, input):
        """
        :param input: [feature, gt]: features: features, shape=NxCxHxW, N is batch_size, C is channels of features, gt, shape=Nx1xHxW, N is batch_size
        :return: dict with the prediction
        """

        features, target = input

        features = self.pred_layer(features)
        pred = self.sample_layer(features)

        if self.training:
            loss = self.reg_criterion(pred=pred, target=target)
            return {'pred': pred, 'loss': loss}
        else:
            return {'pred': pred}

