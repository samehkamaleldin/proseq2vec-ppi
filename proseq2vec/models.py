# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from .util import *
import numpy as np
import pytorch_lightning as pl
from sklearn.metrics import average_precision_score


class ProSeq2VecEncoder(nn.Module):
    def __init__(self, seq_len, aa_k=16, seq_k=100, hidden_size=128, dropout=0.1, nb_layers=2, fc_size=1024,
                 is_bidirectional=False, activation_type="none", *args, **kwargs):
        """ Initialise a new instance of the ProSeqLSTMEncoder class

        Parameters
        ----------
        seq_len: int
            the input sequence fixed length
        aa_k: int
            the size of the amino acid embeddings
        seq_k: int
            the size of the sequence embeddings
        dropout : float
            network dropout weight
        fc_dim: int
            The size of the feedforward layer
        activation : str
            The name of the activation function to use. Options: {"relu", "gelu"}. Otherwise no activation is applied
        """
        super().__init__()
        self.seq_len = seq_len
        self.dropout = dropout
        self.fc_size = fc_size
        self.em_size_aa = aa_k
        self.em_size_seq = seq_k
        self.hidden_size = hidden_size
        self.nb_layers = nb_layers
        self.activation_type = activation_type
        self.dropout_fn = nn.Dropout(dropout)
        self.encoder = torch.nn.Embedding(AminoAcids.get_amino_acids_count() + 1, aa_k, padding_idx=0)
        self.lstm = nn.LSTM(seq_len, hidden_size=hidden_size, bidirectional=is_bidirectional, num_layers=nb_layers, dropout=dropout)
        self.lstm_out_size = self.hidden_size if not is_bidirectional else self.hidden_size * 2
        self.linear_w1 = torch.nn.Linear(self.lstm_out_size * self.em_size_aa, self.fc_size)
        self.linear_w2 = torch.nn.Linear(self.fc_size, self.em_size_seq)

        # activation modules
        self.act_relu = torch.nn.ReLU()
        self.act_gelu = torch.nn.GELU()
        self.act_leaky_relu = torch.nn.LeakyReLU(0.1)

    def forward(self, sequence):
        """ Execute the model pipeline on input data

        Parameters
        ----------
        sequence : torch.Tensor
            A tensor of type int (amino acid code number) and size [b, l] where `b` is the batch size and `l` is the sequence length.

        Returns
        -------
        torch.Tensor
            A tensor of type float and size [b, k] where `b` is the batch size and `k` is the sequence embedding size.
        """
        seq_em = self.encoder(sequence)
        seq_em, _ = self.lstm(seq_em.view([-1, self.em_size_aa, self.seq_len]))
        seq_em = self.linear_w1(seq_em.view([sequence.shape[0], -1]))

        # ------------------------------------------
        # apply activation if specified
        # ------------------------------------------
        if self.activation_type == "relu":
            seq_em = self.act_relu(seq_em)
        elif self.activation_type == "leaky_relu":
            seq_em = self.act_leaky_relu(seq_em)
        elif self.activation_type == "gelu":
            seq_em = self.act_gelu(seq_em)
        # ------------------------------------------

        seq_em = self.dropout_fn(seq_em)
        seq_em = self.linear_w2(seq_em)
        return seq_em


class ProSeq2VecPpi(pl.LightningModule):
    """ PPI prediction model which uses LSTM based encode protein embeddings of amino acid sequences as features.
    The model uses the dot product between the sequence embedding of both pairs as their interaction score.
    """

    def __init__(self, seq_len, aa_k=10, seq_k=128, hidden_size=512, dropout=0.25, nb_layers=2, lr=0.001,
                 is_bidirectional=True, fc_size=1024, activation_type="none", tb_log_mode="debug", *args, **kwargs):
        """ Initialise a new instance of the PPIsModelDotP class

        Parameters
        ----------
        seq_len: int
            the input sequence fixed length
        aa_k: int
            the size of the amino acid embeddings
        seq_k: int
            the size of the sequence embeddings
        dropout : float
            network dropout weight
        activation : str
            Activation type. Options {"relu", "gelu"}, Otherwise nothing happens.
        args: list
            List of other arguments
        kwargs : dict
            Dictionary of other keyed arguments
        """
        super().__init__()
        self.tb_log_mode = tb_log_mode
        self.lr = lr
        self.p2v_encoder = ProSeq2VecEncoder(seq_len, aa_k, seq_k, hidden_size, dropout, nb_layers, activation_type=activation_type, fc_size=fc_size, is_bidirectional=is_bidirectional, **kwargs)
        self.loss_fn = torch.nn.MSELoss(reduction="sum")
        self.decoder = torch.nn.Linear(seq_k, 2)

        self.softmax = nn.Softmax(dim=1)
        self.save_hyperparameters()

    def forward(self, p1_seq, p2_seq):
        """ Predict interactions between pairs of protein sequences

        Parameters
        ----------
        p1_seq : torch.Tensor
            A tensor of type int (amino acid code number) and size [b, l] where `b` is the batch size and `l` is
            the sequence length which represents the first side of the protein pairs.

        p2_seq : torch.Tensor
            A tensor of type int (amino acid code number) and size [b, l] where `b` is the batch size and `l` is
            the sequence length which represents the other side of the protein pairs.
        Returns
        -------
        torch.Tensor
            A tensor of type float and size [b] which represents the scores corresponding to the input sequence pairs
        """

        seq1_embedding = self.p2v_encoder.forward(p1_seq)
        seq2_embedding = self.p2v_encoder.forward(p2_seq)
        embedding_interaction = seq1_embedding * seq2_embedding
        scores = self.decoder(embedding_interaction).reshape([p1_seq.shape[0], 2])
        logits = self.softmax(scores)
        return logits

    def configure_optimizers(self):
        return torch.optim.Adam(params=self.parameters(), lr=self.lr, amsgrad=True)

    def training_step(self, batch, batch_idx):
        (p1_seq, p2_seq), labels = batch
        labels_one_hot = torch.nn.functional.one_hot(labels.long()).type(torch.float32)
        logits = self.forward(p1_seq, p2_seq)
        train_loss = self.loss_fn(logits, labels_one_hot)
        correct = logits.argmax(dim=1).eq(labels).sum().item()

        return {"loss": train_loss, "total": len(labels), "correct": correct,
                'train_scores': logits.data.tolist(), 'train_labels': labels.data.tolist()
                }

    def training_epoch_end(self, outputs):
        scores = np.array([s for p in outputs for s in p['train_scores']])
        labels = np.array([s for p in outputs for s in p['train_labels']])
        train_ap = average_precision_score(labels, scores[:, 1])

        correct = sum([x["correct"] for x in outputs])
        total = sum([x["total"] for x in outputs])
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        return {"loss": avg_loss, "accuracy": correct / total, "ap": train_ap}

    def validation_step(self, batch, batch_idx):
        (p1_seq, p2_seq), labels = batch
        logits = self.forward(p1_seq, p2_seq)
        labels_one_hot = torch.nn.functional.one_hot(labels.long()).type(torch.float32)
        loss = self.loss_fn(logits, labels_one_hot)
        correct = logits.argmax(dim=1).eq(labels).sum().item()

        return {
            'loss': loss,
            'val_scores': logits.data.tolist(),
            'val_labels': labels.data.tolist(),
            "total": len(labels),
            "correct": correct
        }

    def validation_epoch_end(self, outputs):
        scores = np.array([s for p in outputs for s in p['val_scores']])
        labels = np.array([s for p in outputs for s in p['val_labels']])
        scores_1d = scores[:, 1]

        val_ap = average_precision_score(labels, scores_1d)
        correct = sum([x["correct"] for x in outputs])
        total = sum([x["total"] for x in outputs])
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        return {"loss": avg_loss, "accuracy": correct/total, "ap": val_ap}

    def test_step(self, batch, batch_idx):
        (p1_seq, p2_seq), labels = batch
        logits = self.forward(p1_seq, p2_seq)
        labels_one_hot = torch.nn.functional.one_hot(labels.long()).type(torch.float32)
        test_loss = self.loss_fn(logits, labels_one_hot)
        correct = logits.argmax(dim=1).eq(labels).sum().item()
        return {
            'test_loss': test_loss,
            'test_scores': logits.data.tolist(),
            'test_labels': labels.data.tolist(),
            'correct': correct,
            'total': len(labels)
        }

    def test_epoch_end(self, outputs):
        scores = np.array([s for p in outputs for s in p['test_scores']])
        labels = np.array([s for p in outputs for s in p['test_labels']])
        scores_1d = scores[:, 1]

        nb_pos = np.sum(labels)
        nb_neg = labels.shape[0] - np.sum(labels)

        nb_records = labels.shape[0]
        test_ap = average_precision_score(labels, scores_1d)
        test_p10 = precision_at_k(labels, scores_1d, 10) if nb_records >= 10 else -1.
        test_p50 = precision_at_k(labels, scores_1d, 50) if nb_records >= 50 else -1.
        test_p100 = precision_at_k(labels, scores_1d, 100) if nb_records >= 100 else -1.
        test_p500 = precision_at_k(labels, scores_1d, 500) if nb_records >= 500 else -1.
        test_p1000 = precision_at_k(labels, scores_1d, 1000) if nb_records >= 1000 else -1.

        test_h10 = hits_at_k_score(labels, scores_1d, 10) if nb_records >= 10 else -1.
        test_h50 = hits_at_k_score(labels, scores_1d, 50) if nb_records >= 50 else -1.
        test_h100 = hits_at_k_score(labels, scores_1d, 100) if nb_records >= 100 else -1.
        test_h500 = hits_at_k_score(labels, scores_1d, 500) if nb_records >= 500 else -1.
        test_h1000 = hits_at_k_score(labels, scores_1d, 1000) if nb_records >= 1000 else -1.

        correct = sum([x["correct"] for x in outputs])
        total = sum([x["total"] for x in outputs])
        test_acc = correct/total

        metrics = {"ap": test_ap, "accuracy": test_acc,
                   "hits@10": test_h10, "hits@50": test_h50, "hits@100": test_h100, "hits@500": test_h500,
                   "hits@1000": test_h1000, "p@10": test_p10, "p@50": test_p50, "p@100": test_p100, "p@500": test_p500,
                   "p@1000": test_p1000, "positives": nb_pos, "negatives": nb_neg, "neg2pos": nb_neg/nb_pos}

        return {"metrics": metrics, "logits": scores, "labels": labels}
