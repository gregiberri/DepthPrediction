# -*- coding: utf-8 -*-
# @Time    : 2019/12/26 下午3:20
# @Author  : Wang Xin
# @Email   : wangxin_buaa@163.com
# @File    : build.py

import numpy as np
import torch
import re
from torch._six import container_abcs, string_classes, int_classes

from torch.utils.data import DataLoader

from ml.utils.pyt_ops import interpolate

np_str_obj_array_pattern = re.compile(r'[SaUO]')

default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")


# reimplementation to make different target have same shape.
def collate_fn(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        #### start
        h, w = elem.shape[-2:]
        if len(batch) > 0:
            for i in range(1, len(batch)):
                th, tw = batch[i].shape[-2:]
                h, w = min(h, th), min(w, tw)
        for i in range(len(batch)):
            if batch[i].shape[-2:] != (h, w):
                batch[i] = interpolate(batch[i], size=(h, w), mode="nearest")
        #### end
        out = None
        # if torch.utils.data.get_worker_info() is not None:
        #     # If we're in a background process, concatenate directly into a
        #     # shared memory tensor to avoid an extra copy
        #     numel = sum([x.numel() for x in batch])
        #     storage = elem.storage()._new_shared(numel)
        #     out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return collate_fn([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return {key: collate_fn([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(collate_fn(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        transposed = zip(*batch)
        return [collate_fn(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))


