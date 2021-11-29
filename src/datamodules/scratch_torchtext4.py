
import torchtext
import torch
from torchtext.datasets import IMDB, AG_NEWS
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab, build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import GloVe
import torch.nn as nn
import numpy as np


tokenizer = get_tokenizer('basic_english')

train_iter = AG_NEWS(split='train')

def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

emb_dim = 100

# get pretrained glove vectors
glove = GloVe(name = '6B',
              dim = emb_dim)

# create a tensor used for holding the pre-trained vectors for each element of the vocab
pretrained_embedding = torch.zeros(len(vocab), emb_dim)

# get the pretrained vector's vocab, Dict[str, int]
# pretrained_vocab = glove.vectors.get_stoi()

pretrained_vocab = glove.stoi

for idx, token in enumerate(vocab.get_itos()):
    if token in pretrained_vocab:
        pretrained_vector = glove[token] # pretrained_vector is a FloatTensor pre-trained vector for `token`
        pretrained_embedding[idx] = pretrained_vector # update the appropriate row in pretrained_embedding
print(pretrained_embedding[1])






class SentimentLSTM(nn.Module):
    
    def __init__(self, n_vocab, n_embed, n_hidden, n_output, n_layers, drop_p = 0.8):
        super().__init__()
        
        self.n_vocab = n_vocab  
        self.n_layers = n_layers 
        self.n_hidden = n_hidden 
        
        self.embedding = nn.Embedding(n_vocab, n_embed)
        self.lstm = nn.LSTM(n_embed, n_hidden, n_layers, batch_first = True, dropout = drop_p)
        self.dropout = nn.Dropout(drop_p)
        self.fc = nn.Linear(n_hidden, n_output)
        self.sigmoid = nn.Sigmoid()
        
        
    def forward (self, input_words):
                          
        embedded_words = self.embedding(input_words)
        lstm_out, h = self.lstm(embedded_words) 
        lstm_out = self.dropout(lstm_out)
        lstm_out = lstm_out.contiguous().view(-1, self.n_hidden)
        fc_out = self.fc(lstm_out)                  
        sigmoid_out = self.sigmoid(fc_out)              
        sigmoid_out = sigmoid_out.view(batch_size, -1)  
        
        sigmoid_last = sigmoid_out[:, -1]
        
        return sigmoid_last, h
    
    
    def init_hidden (self, batch_size):
        
        device = "cpu"
        weights = next(self.parameters()).data
        h = (weights.new(self.n_layers, batch_size, self.n_hidden).zero_().to(device),
             weights.new(self.n_layers, batch_size, self.n_hidden).zero_().to(device))
        
        return h



nb_samples = 100
features = torch.randint(0, 10000, (nb_samples, 2))
labels = torch.empty(nb_samples, dtype=torch.long).random_(10)

dataset = torch.utils.data.TensorDataset(features, labels)
loader = DataLoader(
    dataset,
    batch_size=2
)


def pad_text(txt, seq_length):
  if len(txt) >= seq_length:
    res = txt[:seq_length]
  else:
    res = ['']*(seq_length-len(txt)) + txt
        
  return res


def convert_unknowns(list_input, unknown_index):
    
  return list(map(lambda x: unknown_index if x=="" else x, list_input))

def tpipeline(x,seq_length, unknown_index=0):
  res = vocab(tokenizer(x))
  res = pad_text(res, seq_length)
  res = map(lambda x: 0 if x=="" else x, res)
  return list(res)

def tpipeline_testing(x,seq_length, unknown_index=0):

  res = pad_text(x, seq_length)
  res = list(map(lambda x: unknown_index if x=="" else x, res))
  return res

text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: x.long()

class MyCollator(object):
    def __init__(self, seq_length):
        self.seq_length = seq_length
    def __call__(self, batch):
      label_list, text_list = [], []
      for (_label, _text) in batch:
          label_list.append(label_pipeline(_label))
          processed_text = torch.tensor(tpipeline(_text, self.seq_length), dtype=torch.int64)
          text_list.append(processed_text)
      label_list = torch.tensor(label_list, dtype=torch.int64)
      text_list = torch.tensor(text_list, dtype=torch.float32 )

      return text_list, label_list


class MyOtherCollator(object):
    def __init__(self, seq_length, unknown_index=0):
        self.seq_length = seq_length
        self.unknown_index = unknown_index

    def __call__(self, batch):
      text, label = list(zip(*batch))
      texts = list(map(lambda x: x.tolist(), text))
      #texts = list(map(lambda x: vocab(tokenizer(x)), texts))
      texts = list(map(lambda x: pad_text(x, self.seq_length), texts))
      texts = list(map(lambda x: convert_unknowns(x, self.unknown_index), texts))
      ttexts = list(map(lambda x: torch.LongTensor(x), texts))      
    
      return torch.stack(ttexts), torch.stack(label)

collate_batch = MyOtherCollator(5, 0)
dataloader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate_batch)

dl_without_collate = DataLoader(dataset, batch_size=8, shuffle=False)


print(next(iter(dataloader)))
print(next(iter(dl_without_collate)))