

from typing import Optional, Tuple

import torchtext
import torch
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab, build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import GloVe
import torch.nn as nn
import numpy as np
import copy
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
import pytorch_lightning as pl




class AGNewsDataModule(pl.LightningDataModule):
    def __init__(self, 
                 batch_size,
                 sequence_length, 
                 num_workers=0,
                 num_classes=4,
                 unknown_index=0,
                 pin_memory=True):

        super().__init__()
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.unknown_index = unknown_index
        self.tokenizer = get_tokenizer('basic_english')


    def prepare_data(self):

        train_iter, self.test_iter = AG_NEWS()
        lengths = [int(len(train_iter)*0.8), int(len(train_iter)*0.2)]
        train_iter, val_iter = random_split(train_iter, lengths)
        self.train_iter = train_iter.dataset
        self.val_iter = val_iter.dataset


    def setup(self, stage: Optional[str] = None):

        self.vocab = build_vocab_from_iterator(self.yield_tokens(self.train_iter), 
                                            specials=["<unk>"])
        self.vocab.set_default_index(self.vocab["<unk>"])


        train_iter, self.test_iter = AG_NEWS()
        lengths = [int(len(train_iter)*0.8), int(len(train_iter)*0.2)]
        train_iter, val_iter = random_split(train_iter, lengths)
        self.train_iter = train_iter.dataset
        self.val_iter = val_iter.dataset


        self.collation_function = MyCollator(self.vocab, 
                                            self.sequence_length, 
                                            self.unknown_index)

    def train_dataloader(self):
        return DataLoader(
            self.train_iter,
            batch_size=self.batch_size,
            drop_last=True,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            collate_fn=self.collation_function
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_iter,
            batch_size=self.batch_size,
            drop_last=True,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            collate_fn=self.collation_function
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_iter,
            batch_size=self.batch_size,
            drop_last=True,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            collate_fn=self.collation_function
        )


    def yield_tokens(self, data_iter):
        for _, text in data_iter:
            yield self.tokenizer(text)


class MyCollator(object):
    def __init__(self, vocab, seq_length, unknown_index=0):
        self.vocab = vocab
        self.seq_length = seq_length
        self.unknown_index = unknown_index
        self.tokenizer = get_tokenizer('basic_english')

    def __call__(self, batch):
      label, text = list(zip(*batch))
      label = list(map(lambda x: x - 1, label))
      #texts = list(map(lambda x: x.tolist(), text))
      texts = list(map(lambda x: self.vocab(self.tokenizer(x)), text))
      texts = list(map(lambda x: self.pad_text(x), texts))
      texts = list(map(lambda x: self.convert_unknowns(x), texts))
      ttexts = list(map(lambda x: torch.LongTensor(x), texts))      
    
      return torch.stack(ttexts), torch.LongTensor(label)

    def pad_text(self, txt):
        if len(txt) >= self.seq_length:
            res = txt[:self.seq_length]
        else:
            res = ['']*(self.seq_length-len(txt)) + txt
            
        return res

    def convert_unknowns(self, list_input):
        
        return list(map(lambda x: self.unknown_index if x=="" else x, list_input))




if __name__ == '__main__':
    t = AGNewsDataModule(10, 7)
    t.prepare_data()
    t.setup()
    ds = t.train_dataloader()
    print(next(iter(ds)))
    ds = t.val_dataloader()
    print(next(iter(ds)))
    ds = t.test_dataloader()
    print(next(iter(ds)))
