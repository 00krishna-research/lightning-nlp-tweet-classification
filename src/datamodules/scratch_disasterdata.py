import collections
import numpy as np
import pandas as pd
import re
import spacy
from torchtext.legacy.data import Field
from torchtext.data import Field, TabularDataset, BucketIterator
from typing import Optional, Tuple
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
import string
from typing import Optional
import pandas as pd
import pytorch_lightning as pl


TRAINPCT = 0.8
TESTPCT = 0.2

# Read raw data
tweets = pd.read_csv("data/train.csv", header=0)

# Create training/test/validation split
def train_test_val_split(x, trainpct=0.8):
    r1, r2 = np.random.random(2)
    if r1 <= trainpct:
        res = "train"
    else:
        res = "test"

    if res=="train" and r2 <= trainpct:
        res = "train"
    elif res=="train" and r2 > trainpct:
        res = "val"
    else:
        pass
    return res

# Preprocess the reviews
def preprocess_text(text):
    text = ' '.join(word.lower() for word in text.split(" "))
    text = re.sub(r"([.,!?])", r" \1 ", text)
    text = re.sub(r"[^a-zA-Z.,!?]+", r" ", text)
    return text

tweets["split"] = 0
tweets["split"] = tweets.split.apply(train_test_val_split)
tweets["text"] = tweets.text.apply(preprocess_text)

# Write munged data to CSV
tweets.to_csv("output.csv", index=False)

spacy_en = spacy.load("en")

def tokenize(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


quote = Field(sequential=True, use_vocab=True, tokenize=tokenize, lower=True)
score = Field(sequential=False, use_vocab=False)

fields = {"quote": ("q", quote), "score": ("s", score)}

train_data, test_data = TabularDataset.splits(
    path="mydata", train="train.json", test="test.json", format="json", fields=fields
)