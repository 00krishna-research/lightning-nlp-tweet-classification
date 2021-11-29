import torchtext
import torch





from torchtext.datasets import IMDB
train_iter, test_iter = IMDB(split=('train', 'test'))

from torchtext.data.utils import get_tokenizer
tokenizer = get_tokenizer('basic_english')

from collections import Counter
from torchtext.vocab import Vocab

train_iter = IMDB(split='train')
counter = Counter()
for (label, line) in train_iter:
    counter.update(tokenizer(line))
vocab = Vocab(counter, min_freq=10, specials=('<unk>', '<BOS>', '<EOS>', '<PAD>'))

print("The length of the new vocab is", len(vocab))
new_stoi = vocab.stoi
print("The index of '<BOS>' is", new_stoi['<BOS>'])
new_itos = vocab.itos
print("The token at index 2 is", new_itos[2])


text_transform = lambda x: [vocab['<BOS>']] + [vocab[token] for token in tokenizer(x)] + [vocab['<EOS>']]
label_transform = lambda x: 1 if x == 'pos' else 0

# Print out the output of text_transform
print("input to the text_transform:", "here is an example")
print("output of the text_transform:", text_transform("here is an example"))

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


def collate_batch(batch):
    label_list, text_list = [], []
    for (_label, _text) in batch:
        label_list.append(label_transform(_label))
        processed_text = torch.tensor(text_transform(_text))
        text_list.append(processed_text)
    return torch.tensor(label_list), pad_sequence(text_list, padding_value=3.0)

train_iter = IMDB(split='train')
train_dataloader = DataLoader(list(train_iter), batch_size=8, shuffle=True, 
                              collate_fn=collate_batch)


print(next(iter(train_dataloader)))



