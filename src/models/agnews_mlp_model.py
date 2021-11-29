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




class AGNewsClassifierMLP(pl.LightningModule):

    def __init__(self, 
                num_classes,
                embedding_dim,
                num_embeddings, 
                dropout_p, 
                # pretrained_embeddings,
                max_seq_length, 
                learning_rate, 
                weight_decay,
                padding_idx=0):
        super().__init__()

        self.save_hyperparameters()

        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.dropout_p = dropout_p
        # self.pretrained_embeddings = pretrained_embeddings
        self.max_seq_length = max_seq_length
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.padding_idx = padding_idx

        # embedding_dim = pretrained_embeddings.size(1)
        # num_embeddings = pretrained_embeddings.size(0)



        self.emb = torch.nn.Embedding(
                                      num_embeddings=self.num_embeddings + 1,
                                      embedding_dim=self.embedding_dim,
                                    #   embedding_dim=embedding_dim,
                                    #   num_embeddings=num_embeddings,
                                      padding_idx=self.padding_idx,
                                      #_weight=pretrained_embeddings
                                      )
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
        x, y = batch
        y_pred = self(x)
        loss = self.loss(y_pred, y).mean()
        self.log('loss', loss, prog_bar=True, on_step=False,
                 on_epoch=True)
        if batch_idx == 1874:
            print("here")
        return {'loss': loss, 'log': {'train_loss': loss}}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss(y_pred, y)
        acc = (y_pred.argmax(-1) == y).float()
        self.log('val/acc', acc, prog_bar=True, on_step=True,
                 on_epoch=True)
        return {'val/loss': loss, 'val/acc': acc}

    # def validation_epoch_end(self, outputs):
    #     loss = torch.cat([o['val/loss'] for o in outputs], 0).mean()
    #     acc = torch.cat([o['val/acc'] for o in outputs], 0).mean()
    #     out = {'val/loss': loss, 'val/acc': acc}
    #     return {**out, 'log': out}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),
                                lr=self.hparams.learning_rate,
                                weight_decay=self.hparams.weight_decay)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss(y_pred, y)
        acc = (y_pred.argmax(-1) == y).float()
        return {'test/loss': loss, 'test/acc': acc}

    def test_epoch_end(self, outputs):
        loss = torch.cat([o['loss'] for o in outputs], 0).mean()
        acc = torch.cat([o['acc'] for o in outputs], 0).mean()
        out = {'test/loss': loss, 'test/acc': acc}
        return {**out, 'log': out}

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


if __name__ == "__main__":
    m = AGNewsClassifierMLP(4, 95811, 20, 0.70, 20, 0.001, 1e-5, 0)
    d = torch.randint(0, 95811, (1, 20))
    m(d)