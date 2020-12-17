#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 2020-05-02 18:33
@Author  : Wang Xin
@Email   : wangxin_buaa@163.com
@File    : base_model.py
"""
import numpy as np

from ml.visualizers.utils import depth_to_color, error_to_color
from ml.utils.pyt_ops import tensor2numpy, interpolate


class Visualizer:
    def __init__(self, writer=None):
        self.writer = writer

    def visualize(self, batch, out, epoch, tag=''):
        """
            :param batch: minibatch
            :param out: model output for visualization, dic, {"target": [NxHxW]}
            :return: vis_ims: image for visualization.
        """

        fn = batch["fn"]
        if batch["target"].shape != out[-1].shape:
            h, w = batch["target"].shape[-2:]
            # batch = interpolate(batch, size=(h, w), mode='nearest')
            out = interpolate(out, size=(h, w), mode='bilinear', align_corners=True)

        images = batch["image_n"].numpy()
        depths = batch['depth_n'].numpy()

        has_gt = False
        if batch.get("target") is not None:
            depth_gts = tensor2numpy(batch["target"])
            has_gt = True

        for i in range(len(fn)):
            image = images[i].astype(np.float)
            depth = depths[i].astype(np.float)
            pred_depth = tensor2numpy(out[i]).squeeze()
            # print("!! depth shape:", depth.shape)

            if has_gt:
                depth_gt = depth_gts[i].squeeze()

                err = error_to_color(pred_depth, depth_gt)
                depth_gt = depth_to_color(depth_gt)

            depth = depth_to_color(depth)
            pred_depth = depth_to_color(pred_depth)
            # print("pred:", depth.shape, " target:", depth_gt.shape)
            group = np.concatenate((depth, image, pred_depth), axis=0)

            if has_gt:
                gt_group = np.concatenate((err, depth_gt, depth_gt), axis=0)
                group = np.concatenate((group, gt_group), axis=1)

            if self.writer is not None:
                group = group.transpose((2, 0, 1)) / 255.0
                group = group.astype(np.float32)
                # print("group shape:", group.shape)
                self.writer.add_image(fn[i] + f"{tag}/image", group, epoch)

