import torch
import numpy as np

from ml.modules.losses.ordinal_regression_loss import OrdinalRegressionLoss

config_path = '../config/config_files/kitti_base.yaml'

ord_num = 90
gamma = -0.97
beta = 90
config = load_config(config_path)
config['model']['params']['discretization'] = "SID"

ordinal_regression_loss = OrdinalRegressionLoss(ord_num, beta)

val_loader, niter_val = build_loader(config, is_train=False)

rmses = []

for i, data in enumerate(val_loader):
    gt = data['target'].unsqueeze(0)

    _, gt_mask = ordinal_regression_loss._create_ord_label(gt)

    label = ord_num - torch.sum(gt_mask, dim=1)
    t0 = torch.exp(np.log(beta) * label.float() / ord_num)
    t1 = torch.exp(np.log(beta) * (label.float() + 1) / ord_num)

    depth_gt = (t0 + t1) / 2 - gamma

    depth_gt = np.squeeze(depth_gt.numpy())
    gt = np.squeeze(gt.numpy())

    gt_mask = gt > 0
    gt = gt[gt_mask]
    depth_gt = depth_gt[gt_mask]

    rmse = np.sqrt(np.mean((depth_gt - gt) ** 2)) * 1000

    rmses.append(rmse)
    print(f'{i + 1} / {len(val_loader)}, RMSE: {rmse}')

mean_rmse = np.mean(rmses)
print(f'Mean minimum RMSE: {mean_rmse}')
