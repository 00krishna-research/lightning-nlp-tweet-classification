from typing import Any, List

import torch
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric
from torchmetrics.classification.accuracy import Accuracy


from argparse import ArgumentParser

import sh
import torch.utils.data
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
import pytorch_lightning as pl




class DisasterTweetsClassifierMLP(pl.LightningModule):

    def __init__(self, num_classes, dropout_p, pretrained_embeddings,
                 max_seq_length, learning_rate, weight_decay):
        super().__init__()

        self.save_hyperparameters('num_classes', 'dropout_p', 'learning_rate', 'weight_decay')

        embedding_dim = pretrained_embeddings.size(1)
        num_embeddings = pretrained_embeddings.size(0)

        self.emb = torch.nn.Embedding(embedding_dim=embedding_dim,
                                      num_embeddings=num_embeddings,
                                      padding_idx=0,
                                      _weight=pretrained_embeddings)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=embedding_dim * max_seq_length, out_features=128),
            torch.nn.BatchNorm1d(num_features=128),
            torch.nn.Dropout(p=dropout_p),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=128, out_features=64),
            torch.nn.BatchNorm1d(num_features=64),
            torch.nn.Dropout(p=dropout_p),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=64, out_features=32),
            torch.nn.BatchNorm1d(num_features=32),
            torch.nn.Dropout(p=dropout_p),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=32, out_features=16),
            torch.nn.BatchNorm1d(num_features=16),
            torch.nn.Dropout(p=dropout_p),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=16, out_features=num_classes)
        )
        self.loss = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, batch):
        x_embedded = self.emb(batch).view(batch.size(0), -1)
        output = self.fc(x_embedded)

        return output

    def training_step(self, batch, batch_idx):
        y_pred = self(batch['x_data'])
        loss = self.loss(y_pred, batch['y_target']).mean()
        return {'loss': loss, 'log': {'train_loss': loss}}

    def validation_step(self, batch, batch_idx):
        y_pred = self(batch['x_data'])
        loss = self.loss(y_pred, batch['y_target'])
        acc = (y_pred.argmax(-1) == batch['y_target']).float()
        return {'loss': loss, 'acc': acc}

    def validation_epoch_end(self, outputs):
        loss = torch.cat([o['loss'] for o in outputs], 0).mean()
        acc = torch.cat([o['acc'] for o in outputs], 0).mean()
        out = {'val_loss': loss, 'val_acc': acc}
        return {**out, 'log': out}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),
                                lr=self.hparams.learning_rate,
                                weight_decay=self.hparams.weight_decay)

    def test_step(self, batch, batch_idx):
        y_pred = self(batch['x_data'])
        loss = self.loss(y_pred, batch['y_target'])
        acc = (y_pred.argmax(-1) == batch['y_target']).float()
        return {'loss': loss, 'acc': acc}

    def test_epoch_end(self, outputs):
        loss = torch.cat([o['loss'] for o in outputs], 0).mean()
        acc = torch.cat([o['acc'] for o in outputs], 0).mean()
        out = {'test_loss': loss, 'test_acc': acc}
        return {**out, 'log': out}

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--dropout_p', default=0.7, type=float)
        parser.add_argument('--learning_rate', default=1e-3, type=float)
        parser.add_argument('--weight_decay', default=1e-5, type=float)
        return parser


def predict_target(text, classifier, vectorizer, max_seq_length):
    text_vector, _ = vectorizer.vectorize(text, max_seq_length)
    text_vector = torch.tensor(text_vector)
    pred = torch.nn.functional.softmax(classifier(text_vector.unsqueeze(dim=0)), dim=1)
    probability, target = pred.max(dim=1)

    return {'pred': target.item(), 'probability': probability.item()}


def predict_on_dataset(classifier, ds):
    classifier.eval()
    df = pd.DataFrame(columns=["text", "target", "pred", "probability"])
    for sample in iter(ds):
        result = predict_target(sample['text'], classifier, ds.get_vectorizer(), ds.get_max_seq_length())
        result['target'] = sample['y_target'].item()
        result['text'] = sample['text']
        df = df.append(result, ignore_index=True)
    df.target = df.target.astype(np.int32)
    df.pred = df.pred.astype(np.int32)

    f1 = f1_score(df.target, df.pred)
    acc = accuracy_score(df.target, df.pred)
    roc_auc = roc_auc_score(df.target, df.probability)
    print("Result metrics - \n Accuracy={} \n F1-Score={} \n ROC-AUC={}".format(acc, f1, roc_auc))
    return df
