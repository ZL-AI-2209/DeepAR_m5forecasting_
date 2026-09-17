
from __future__ import division
import numpy as np
import torch
import os
import logging
from torch.utils.data import DataLoader, Dataset, Sampler

logger = logging.getLogger('DeepAR.Data')

import torch
from torch.utils.data import Dataset, DataLoader
import zarr
import numpy as np


import torch
from torch.utils.data import Dataset
import zarr




class LoadDataset(Dataset):
    def __init__(self, zarr_path, transform=None):
        self.zarr_path = zarr_path
        self.transform = transform

        temp_root = zarr.open(zarr_path, mode='r')
        self.length = temp_root['x_input'].shape[0]
        self.root = None

    def _open_zarr(self):
        if self.root is None:
            self.root = zarr.open(self.zarr_path, mode='r')
            self.x_arr = self.root['x_input']
            self.labels_arr = self.root['labels']
            self.v_arr = self.root['v_input']
            self.static_arr = self.root['series_id']
            self.mask_labels_arr = self.root['labels_mask']

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        self._open_zarr()

        x_input = self.x_arr[idx]
        labels = self.labels_arr[idx]
        v_input = self.v_arr[idx]
        static_cov = self.static_arr[idx]
        mask_val = self.mask_labels_arr[idx]

        x_tensor = torch.from_numpy(np.ascontiguousarray(x_input)).to(torch.float32)
        labels_tensor = torch.from_numpy(np.ascontiguousarray(labels)).to(torch.float32)
        v_tensor = torch.from_numpy(np.ascontiguousarray(v_input)).to(torch.float32)
        static_tensor = torch.from_numpy(np.ascontiguousarray(static_cov)).long()
        mask_tensor = torch.from_numpy(np.ascontiguousarray(mask_val)).to(torch.float32)

        if self.transform:
            x_tensor = self.transform(x_tensor)

        return x_tensor, static_tensor, v_tensor, labels_tensor, mask_tensor


class WeightedSampler(Sampler):
    def __init__(self, zarr_path, replacement=True):
        self.zarr_path = zarr_path
        self.replacement = replacement

        root = zarr.open(zarr_path, mode='r')

        v_scale = np.array(root['v_input'][:, 0], dtype=np.float64)

        abs_v = np.abs(v_scale)
        sum_abs_v = np.sum(abs_v)

        if sum_abs_v == 0:
            self.weights = np.ones(len(abs_v), dtype=np.float64) / len(abs_v)
        else:
            self.weights = abs_v / sum_abs_v

        self.num_samples = len(self.weights)
        logger.info(f'Loaded weights from Zarr. Num samples: {self.num_samples}')

    def __iter__(self):
        # NumPy choice генерирует сэмплы значительно быстрее torch.multinomial на CPU
        indices = np.random.choice(
            self.num_samples,
            size=self.num_samples,
            replace=self.replacement,
            p=self.weights
        )
        return iter(indices)

    def __len__(self):
        return self.num_samples