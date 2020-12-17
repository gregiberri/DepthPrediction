#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 2020-05-02 17:55
@Author  : Wang Xin
@Email   : wangxin_buaa@163.com
@File    : oridinal_regression_layer.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OrdinalRegression(nn.Module):
    def forward(self, x):
        """
        :param x: NxCxHxW, N is batch_size, C is channels of features
        :return: ord_label is ordinal outputs for each spatial locations , N x 1 x H x W
                 ord prob is the probability of each label, N x OrdNum x H x W
        """
        N, C, H, W = x.size()
        ord_num = C // 2

        x = x.view(-1, 2, ord_num, H, W)
        if self.training:
            prob = F.log_softmax(x, dim=1).view(N, C, H, W)
            ord_prob = F.softmax(x, dim=1)[:, 0, :, :, :]
            ord_label = torch.sum((ord_prob > 0.5), dim=1)
            return prob, ord_label

        ord_prob = F.softmax(x, dim=1)[:, 0, :, :, :]
        ord_label = torch.sum((ord_prob > 0.5), dim=1)
        return {'ord_prob': ord_prob, 'ord_label': ord_label}
