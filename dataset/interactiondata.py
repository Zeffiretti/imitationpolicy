"""
 * File Name : 
 * Author    : ZEFFIRETTI, HESH
 * College   : Beijing Institute of Technology
 * E-Mail    : zeffiretti@bit.edu.cn, hiesh@mail.com
"""

from __future__ import print_function, division
import os
from abc import ABC

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader

# ignore warnings
import warnings

warnings.filterwarnings("ignore")

plt.ion()  # interactive mode


class InteractionData(Dataset, ABC):
    """
    InteractionData
    """

    def __init__(self, file_path, index, time_scale=100):
        """
        Args:
            file_path (string): file storing data (txt)
            index (int): index of data to be loaded
        """
        self.raw_data = np.loadtxt(file_path, dtype=np.float32) * 30
        # show raw data shape
        print("raw data shape: ", self.raw_data.shape)
        # convert raw data to tensor
        self.data = torch.from_numpy(self.raw_data)
        # show data shape
        print("data shape: ", self.data.shape)
        self.idx = index
        self.time_scale = time_scale

    def __len__(self):
        return len(self.data[:, 0]) - self.time_scale

    def __getitem__(self, idx):
        """
        return data with:
            x: from idx t idx+time_scale
            y: [idx+time_scale, self.idx]
        """
        x = self.data[idx:idx + self.time_scale, :]
        y = self.data[idx + self.time_scale, self.idx:self.idx + 1]
        return x, y
