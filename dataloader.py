
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

        # Быстро открываем Zarr, чтобы считать только длину датасета
        temp_root = zarr.open(zarr_path, mode='r')
        # Длину берем по количеству строк в основном массиве x_input
        self.length = temp_root['x_input'].shape[0]

        self.root = None  # Будет открыт лениво внутри __getitem__

    def _open_zarr(self):
        if self.root is None:
            self.root = zarr.open(self.zarr_path, mode='r')
            # Привязываем массивы к свойствам класса для быстрого доступа внутри воркера
            self.x_arr = self.root['x_input']
            self.labels_arr = self.root['labels']
            self.v_arr = self.root['v_input']
            self.static_arr = self.root['series_id']
            self.mask_labels = self.root['labels_mask']

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Инициализируем соединение с Zarr в контексте текущего процесса (воркера)
        self._open_zarr()

        # Читаем данные по индексу напрямую из независимых Zarr-массивов
        x_input = self.x_arr[idx]      # [window_size - 1, total_features]
        labels = self.labels_arr[idx]  # [window_size] (в трейне) или [prediction_length] (в тесте)
        v_input = self.v_arr[idx]      
        static_cov = self.static_arr[idx] # [num_static]
        mask = self.mask_labels

        # Конвертируем в PyTorch тензоры (используем copy() для предотвращения read-only ошибок в NumPy)
        x_tensor = torch.from_numpy(x_input.copy()).to(torch.float32)
        labels_tensor = torch.from_numpy(labels.copy()).to(torch.float32)
        v_tensor = torch.from_numpy(v_input.copy()).to(torch.float32)
        static_tensor = torch.from_numpy(static_cov.copy()).long()
        mask = torch.from_numpy(static_cov.copy()).long()

        if self.transform:
            x_tensor = self.transform(x_tensor)

        return x_tensor, static_tensor, v_tensor, labels_tensor, mask



class WeightedSampler(Sampler):
    def __init__(self, zarr_path, replacement=True):

        self.zarr_path = zarr_path
        self.replacement = replacement

        # Открываем Zarr в режиме чтения, чтобы быстро забрать только веса
        root = zarr.open(zarr_path, mode='r')

        # В нашей новой prep_data массив v_input имеет форму [total_windows, 2]
        # v_input[:, 0] — это коэффициент масштабирования (scale) для каждого окна
        v_scale = np.array(root['v_input'][:, 0])

        # Рассчитываем вероятности на основе абсолютных значений масштаба
        abs_v = np.abs(v_scale)
        sum_abs_v = np.sum(abs_v)

        if sum_abs_v == 0:
            # Если все масштабы нулевые (маловероятно), делаем веса равномерными
            self.weights = torch.ones(len(abs_v), dtype=torch.double) / len(abs_v)
        else:
            self.weights = torch.as_tensor(abs_v / sum_abs_v, dtype=torch.double)

        self.num_samples = self.weights.shape[0]

        logger.info(f'Loaded weights from Zarr. Num samples: {self.num_samples}')

    def __iter__(self):
        # Используем multinomial для взвешенного случайного выбора индексов окон
        return iter(torch.multinomial(self.weights, self.num_samples, self.replacement).tolist())

    def __len__(self):
        return self.num_samples