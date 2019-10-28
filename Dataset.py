import os
import torch
import time
import numpy as np
import config as cfg
from torch.utils.data import Dataset
from multiprocessing import Process, cpu_count, Manager

from input_data import data_proc


class onset_dataset(Dataset):
    def __init__(self,wav_files,
                 anno_files,
                 pad_length,
                 spec_style="cqt",
                 dual_channel=False,
                 is_filter=False,
                 is_training=True):
        super(onset_dataset, self).__init__()
        self.pad_length = pad_length
        self.is_filter = is_filter
        self.is_training = is_training

        self.spec_style = spec_style
        self.wav_files = wav_files
        self.anno_files = anno_files
