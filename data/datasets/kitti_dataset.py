#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 2020-05-02 22:28
@Author  : Wang Xin
@Email   : wangxin_buaa@163.com
@File    : Kitti.py
"""
import glob
import os
import numpy as np

from imgaug import HeatmapsOnImage

from data.datasets.base_dataset import BaseDataset
from data.occlusion_filter import filter_occlusions_with_parameters
from data.utils import nomalize, kitti_image_loader, kitti_depth_loader


class Kitti(BaseDataset):
    def __init__(self, config, is_train=True, image_loader=kitti_image_loader, depth_loader=kitti_depth_loader, scenes=''):
        super().__init__(config, is_train, image_loader, depth_loader)
        self.config = config
        self.scenes = scenes
        self.filenames = self._get_filepaths()

    def _get_filepaths(self):
        # for the test split we only have the input image path
        if self.split == 'test':
            # get the input image paths
            image_path_template = os.path.join(*[self.dataset_path, 'kitti_rgb', self.split, '*',
                                                 'image_0[2, 3]', 'data', '*.png'])
            image_paths = glob.glob(image_path_template)
            gt_paths = [None] * len(image_paths)

            return {'image_paths': image_paths, 'gt_paths': gt_paths}
        elif self.split == 'selected_val':
            # get the input image paths
            gt_path_template = os.path.join(*[self.dataset_path,
                                                 'kitti_depth/val_selection_cropped/groundtruth_depth/*.png'])

            gt_paths = glob.glob(gt_path_template)

            def get_image_path_from_gt_path(gt_path):
                image_path = gt_path.replace('groundtruth_depth', 'image')
                if not os.path.exists(image_path):
                    raise ValueError(f'The image pair: {image_path} \n for gt: {gt_path}')

                return image_path

            image_paths = [get_image_path_from_gt_path(gt_path) for gt_path in gt_paths]

            return {'image_paths': image_paths, 'gt_paths': gt_paths}

        # get the gt paths
        if self.scenes != '':
            gt_paths = self.get_files_in_scenes()
        else:
            gt_path_template = os.path.join(*[self.dataset_path, 'kitti_depth', self.split, '*', 'proj_depth',
                                              'groundtruth', 'image_0[2, 3]', '*.png'])
            gt_paths = glob.glob(gt_path_template)

        # use only the first 50 images: for debug
        if self.config.small:
            gt_paths = gt_paths[:50]

        # for train or val get the gt paths from the image paths
        def get_image_path_from_gt_path(gt_path):
            gt_path_parts = gt_path.split('/')
            image_path = os.path.join(*['/', *gt_path_parts[:-7], 'kitti_rgb', *gt_path_parts[-6:-4], gt_path_parts[-2],
                                        'data', gt_path_parts[-1]])
            if not os.path.exists(image_path):
                raise ValueError(f'The image pair: {image_path} \n for gt: {gt_path}')

            return image_path

        def get_depth_path_from_gt_path(gt_path):
            return gt_path.replace('groundtruth', 'velodyne_raw')

        image_paths = [get_image_path_from_gt_path(gt_path) for gt_path in gt_paths]
        depth_paths = [get_depth_path_from_gt_path(gt_path) for gt_path in gt_paths]

        return {'image_paths': image_paths, 'depth_paths': depth_paths, 'gt_paths': gt_paths}

    def _preprocess(self, image, depth, gt):
        crop_h, crop_w = self.config.input_size
        # resize
        H, W = image.shape[:2]
        dH, dW = depth.shape
        gtH, gtW = gt.shape

        assert W == dW and H == dH, \
            "image shape should be same with depth, but image shape is {}, depth shape is {}".format((H, W), (dH, dW))

        # bottom crop
        shape = [crop_h, crop_w]
        image = np.array(image)[H-shape[0]:, W-shape[1]:]
        depth = np.array(depth)[dH-shape[0]:, dW-shape[1]:]
        gt = np.array(gt)[gtH-shape[0]:, gtW-shape[1]:]

        if self.is_train:
            # make heatmap from depth and gt to do only the corresponding augmentations
            depth_gt = np.stack([depth, gt], -1)
            depth_gt_heatmap = HeatmapsOnImage(depth_gt, shape=image.shape, min_value=-1.0, max_value=np.max(depth))

            # augment
            image, depth_gt = self.aug(image=image, heatmaps=depth_gt_heatmap)
            depth, gt = np.transpose(depth_gt.get_arr(), [2, 0, 1])

        # filter out the occluded lidar points
        if self.config.occlusion_filter is not None:
            depth = filter_occlusions_with_parameters(depth,
                                                      self.config.occlusion_filter.params.distance_threshold,
                                                      self.config.occlusion_filter.params.kernel_size)

        # expand dims to get channel first
        depth = np.expand_dims(depth, 0)
        gt = np.expand_dims(gt, 0)

        # normalize
        image_n = np.array(image).astype(np.float32)
        image = nomalize(image_n.copy(), type=self.config.norm_type)

        # transpose to channel first
        image = image.transpose(2, 0, 1)

        # uniform sampling the depth
        uniform_keep_prob = self.lidar_sparsity if self.lidar_sparsity > 0.1 else 0
        mask = np.random.binomial(1, uniform_keep_prob, depth.shape)
        depth[mask == 0] = -1

        output_dict = {"image_n": image_n, 'depth_n': np.squeeze(depth)}

        self.lidar_sparsity *= self.lidar_sparsity_decay

        return image, depth, gt, output_dict

    def get_files_in_scenes(self):
        gt_paths = []
        for scene in self.scenes:
            gt_path_template = os.path.join(*[self.dataset_path, 'kitti_depth', self.split, scene, 'proj_depth',
                                              'groundtruth', 'image_0[2, 3]', '*.png'])
            gt_paths.extend(glob.glob(gt_path_template))

        return gt_paths
