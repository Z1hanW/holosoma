# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import math
import numpy as np
import torch
from torch.utils.data import Dataset


class CriticDataset(Dataset):
    def __init__(self, batch_size, obs, target_values, shuffle=False, drop_last=False):
        self.obs = obs.view(-1, obs.shape[-1])
        self.target_values = target_values.view(-1)
        self.batch_size = batch_size

        if shuffle:
            self.shuffle()

        if drop_last:
            self.length = self.obs.shape[0] // self.batch_size
        else:
            self.length = ((self.obs.shape[0] - 1) // self.batch_size) + 1

    def shuffle(self):
        index = np.random.permutation(self.obs.shape[0])
        self.obs = self.obs[index, :]
        self.target_values = self.target_values[index]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        assert index < self.length, f"{index} {self.length}"
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, self.obs.shape[0])
        return {
            "obs": self.obs[start_idx:end_idx, :],
            "target_values": self.target_values[start_idx:end_idx],
        }


class GeneralizedDataset(Dataset):
    def __init__(self, batch_size, *tensors, shuffle=False):
        assert len(tensors) > 0
        lengths = {len(t) for t in tensors}
        assert len(lengths) == 1

        self.num_tensors = next(iter(lengths))
        self.batch_size = batch_size
        assert (
            self.num_tensors % self.batch_size == 0
        ), f"{self.num_tensors} {self.batch_size}"
        self.tensors = tensors
        self.do_shuffle = shuffle

        if shuffle:
            self.shuffle()

    def shuffle(self):
        index = np.random.permutation(self.tensors[0].shape[0])
        self.tensors = [t[index] for t in self.tensors]

    def __len__(self):
        return self.num_tensors // self.batch_size

    def __getitem__(self, index):
        assert index < len(self), f"{index} {len(self)}"
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, self.num_tensors)
        return [t[start_idx:end_idx] for t in self.tensors]
