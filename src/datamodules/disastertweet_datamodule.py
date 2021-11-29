from typing import Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms


import string
from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.utils.data
from torch.utils.data import Dataset


class DisasterTweetsDataModule(pl.LightningDataModule):
    def __init__(self, 
                 tweets_data_path,
                 batch_size, 
                 num_workers=0,
                 pin_memory=True):

        super().__init__()
        self.num_workers = num_workers
        self.tweets_data_path = tweets_data_path
        self.embeddings_path = embeddings_path
        self.batch_size = batch_size
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None):
        tweets_df = pd.read_csv(self.tweets_data_path)


    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers
        )
