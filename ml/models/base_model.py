import numpy as np
import torch
import torch.nn as nn

from ml.modules.backbones import Backbone
from ml.modules.bottoms import Bottom
from ml.modules.heads import Head
from ml.modules.layers.bifpn import BiFpn
from ml.modules.tops import Top


class BaseModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # bottom
        self.bottom = Bottom(config.bottom)  # conv bn relu

        # get backbone
        self.backbone = Backbone(config.backbone)

        # get top
        self.top = Top(config.top)

        # get head, i wish ;P
        self.head = Head(config.head)

        # if 'dorn' in self.head:
        #     self.dorn_layer = torch.nn.Sequential(torch.nn.Conv2d(in_channels=config['backbone_features'],
        #                                                           out_channels=self.ord_num * 2,
        #                                                           kernel_size=1,
        #                                                           stride=1),
        #                                           OrdinalRegressionLayer())
        #     self.dorn_criterion = OrdinalRegressionLoss(self.ord_num, self.beta, self.discretization)
        # if 'reg' in self.head:
        #     self.reg_layer = torch.nn.Conv2d(in_channels=config['backbone_features'],
        #                                      out_channels=1,
        #                                      kernel_size=1,
        #                                      stride=1)
        #
        #     self.reg_criterion = get_regression_loss(config['regression_loss'])

    def forward(self, image, depth=None, target=None):
        """
        :param image: RGB image, torch.Tensor, Nx3xHxW
        :param target: ground truth depth, torch.Tensor, NxHxW
        :return: output: if training, return loss, torch.Float,
                         else return {"target": depth, "prob": prob, "label": label},
                         depth: predicted depth, torch.Tensor, NxHxW
                         prob: probability of each label, torch.Tensor, NxCxHxW, C is number of label
                         label: predicted label, torch.Tensor, NxHxW
        """
        input_feature = self.bottom(image, depth)

        p0, p1, p2, p3 = self.backbone(input_feature)

        feature = self.top([p0, p1, p2, p3])

        pred = self.head([feature, target])

        return pred

    def get_prediction_and_loss(self, feat, target):
        # predicion
        # dorn prediction
        if 'dorn' in self.head:
            prob, label = self.dorn_layer(feat)
            if self.discretization == "SID":
                t0 = torch.exp(np.log(self.beta) * label.float() / self.ord_num)
                t1 = torch.exp(np.log(self.beta) * (label.float() + 1) / self.ord_num)
            else:
                t0 = 1.0 + (self.beta - 1.0) * label.float() / self.ord_num
                t1 = 1.0 + (self.beta - 1.0) * (label.float() + 1) / self.ord_num
            dorn_depth = (t0 + t1) / 2 - self.gamma
        else:
            dorn_depth = torch.as_tensor([0], device=torch.device('cuda'))

        # regression prediction
        if 'reg' in self.head:
            reg_depth = self.reg_layer(feat).squeeze(1)
        else:
            reg_depth = torch.as_tensor([0], device=torch.device('cuda'))

        # the full depth
        depth = dorn_depth + reg_depth

        # loss
        if self.training and target is not None:
            # dorn loss
            if 'dorn' in self.head:
                dorn_loss = self.dorn_criterion(prob, target)
            else:
                dorn_loss = torch.as_tensor([0], device=torch.device('cuda'))

            # regression loss
            if 'reg' in self.head:
                reg_loss = self.reg_criterion(depth, target)
            else:
                reg_loss = torch.as_tensor([0], device=torch.device('cuda'))

            # full loss
            loss = dorn_loss + reg_loss

        else:
            loss = torch.as_tensor([0], device=torch.device('cuda'))

        return depth, loss
