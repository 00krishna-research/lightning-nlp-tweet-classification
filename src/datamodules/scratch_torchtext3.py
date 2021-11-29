from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer

from typing import Optional, Tuple

import torchtext
import torch
from torchtext.vocab import Vocab, build_vocab_from_iterator
import numpy as np


def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

tokenizer = get_tokenizer('basic_english')
train_iter, test_iter = AG_NEWS()

vocab = build_vocab_from_iterator(yield_tokens(train_iter), 
                                            specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

train_iter, test_iter = AG_NEWS()
print(next(iter(train_iter)))

