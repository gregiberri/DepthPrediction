import os
import time

import numpy as np
import torch

from data.datasets import get_dataloader
from ml.metrics.metrics import Metrics

cwd = os.getcwd()
os.chdir(os.path.join('/', *cwd.split('/')[:-1]))

from config import ConfigNameSpace

config = ConfigNameSpace('config/config_files/kitti_base.yaml')
config.data.params.batch_size = 1
train_loader, niter_per_epoch = get_dataloader(config.data, is_train=True)

raw_point_numbers = []
train_metric = Metrics('', tag='train', niter=niter_per_epoch)

start_time = time.time()
for i, inputs in enumerate(train_loader):
    load_time = time.time() - start_time
    depth = inputs['depth']
    gt = inputs['target']

    raw_mask = torch.logical_or(depth <= 0, gt <= 0)
    inputs['depth'][raw_mask] = -1
    inputs['target'][raw_mask] = -1

    train_metric.compute_metric(inputs['depth'],
                                inputs,
                                inputs['scene'])
    raw_point_numbers.append(np.prod(raw_mask.size()) - np.sum(raw_mask.numpy()))

    print(f'{i}/{len(train_loader)}, '
          f'raw_error: {train_metric.rmse.vals:.3f}, '
          f'raw_error_running_mean: {train_metric.rmse.mean():.3f}, '
          f'raw_point_number: {raw_point_numbers[-1]}, '
          f'time: {load_time: .5f}s')

    start_time = time.time()

print(f'\n'
      f'mean_raw_error: {train_metric.rmse.mean():.3f}\n'
      f'mean_raw_point_number: {np.mean(raw_point_numbers):.3f}')
train_metric.rmse_scene_meter.draw_histogram('6_occlusion_filtered.png')
