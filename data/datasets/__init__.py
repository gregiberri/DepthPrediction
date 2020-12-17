from torch.utils.data import DataLoader

from data.datasets import kitti_dataset
import numpy as np

from data.collate import collate_fn


def get_dataloader(data_config, is_train):
    # get the iterator object
    if data_config.name == 'kitti':
        dataset = kitti_dataset.Kitti(data_config.params, is_train=is_train)
    elif data_config.name == 'audi':
        raise NotImplementedError()
    else:
        raise ValueError(f'Wrong dataset name: {data_config.name}')

    # calculate the iteration number for the tqdm
    batch_size = data_config.params.batch_size if is_train else 1
    niters_per_epoch = int(np.ceil(dataset.get_length() // batch_size))
    shuffle = is_train
    collate_function = collate_fn if is_train else None

    # make the torch dataloader object
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        num_workers=0,
                        drop_last=False,
                        shuffle=shuffle,
                        pin_memory=False,
                        collate_fn=collate_function)

    return loader, niters_per_epoch
