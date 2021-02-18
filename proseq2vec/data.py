# -*- coding: utf-8 -*-

from collections import Iterable, Counter
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from gzip import GzipFile
from io import TextIOWrapper
from .util import AminoAcids


def load_ppi_source(ppi_source, delimiter="\t"):
    if isinstance(ppi_source, str):
        ppi_data = np.loadtxt(ppi_source, delimiter="\t", dtype=np.str)
    elif isinstance(ppi_source, PpiSeqDataset):
        ppi_data = ppi_source.original_ppi
    elif isinstance(ppi_source, np.ndarray):
        ppi_data = ppi_source
    elif isinstance(ppi_source, Subset):
        ppi_data = ppi_source.dataset.original_ppi
    elif isinstance(ppi_source, TextIOWrapper) or isinstance(ppi_source, GzipFile):
        ppi_data = np.array([line.strip().split(delimiter) for line in ppi_source])
    else:
        raise ValueError(f"Unknown ppi_source type ({type(ppi_source)}).")
    return ppi_data


class PpiSeqDataset(Dataset):

    def __init__(self, ppi_source, seq_filepath, max_len, fixed_label=1.0, neg2pos=None, negatives_source=None,  delimiter="\t", device=None):
        """ Initialize a positive instances based protein protein interaction dataset.

        Parameters
        ----------
        ppi_source: object
            ppi source which can be in different forms such as:
                - filepath: `str`
                - file descriptor: `io.TextIOWrapper`
                - gzip compressed file: `gzip.GzipFile`
                - a ppi numpy array: `np.ndarray`.
                - another ppi dataset: `PpiSeqDataset`
                - a subset dataset of another ppi dataset: `Subset`
        seq_filepath: str
            sequence filepath
        max_len: int
            maximum allowed sequence length
        fixed_label: float
            fixed data label to be used as labels for the ppi data when labels are absent
        neg2pos: int
            number of negatives to generate per positive
        device: torch.device
            the torch device to use
        """
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_len = max_len
        self.neg2pos = neg2pos

        # -----------------------------------------------------------------
        # load protein-protein interactions codes from the ppi source
        # -----------------------------------------------------------------
        self.original_ppi = load_ppi_source(ppi_source, delimiter=delimiter)
        self.neg_ppi_source = load_ppi_source(negatives_source, delimiter=delimiter) if negatives_source is not None else None
        if self.neg_ppi_source is not None and self.neg_ppi_source.shape[1] == 3:
            self.neg_ppi_source = self.neg_ppi_source[:, :2]
        # -----------------------------------------------------------------
        self.original_data_len = self.original_ppi.shape[0]
        if neg2pos is not None:
            self.idx_step = neg2pos+1
            self.nb_samples = self.original_data_len * self.idx_step
        else:
            self.nb_samples = self.original_data_len

        self.p1 = np.full([self.nb_samples, ], "#", dtype=np.dtype('U10'))
        self.p2 = np.full([self.nb_samples, ], "#", dtype=np.dtype('U10'))
        self.labels = np.zeros([self.nb_samples, ], dtype=np.float32)

        self.p1[:self.original_data_len] = self.original_ppi[:, 0]
        self.p2[:self.original_data_len] = self.original_ppi[:, 1]
        self.labels[:self.original_data_len] = fixed_label

        self._protein_codes = np.unique(self.original_ppi[:, :2])
        self.protein_codes_set = set(self._protein_codes)
        self.nb_proteins = self._protein_codes.shape[0]
        if self.original_ppi.shape[1] == 3 and (neg2pos is None or neg2pos == 0):
            self.labels = self.original_ppi[:, 2].astype(np.float32)
            self.labels_tensor = torch.from_numpy(self.original_ppi[:, 2].astype(np.float32)).type(torch.float32).to(self.device)
        elif self.original_ppi.shape[1] == 2 or neg2pos is not None:
            self.labels_tensor = torch.from_numpy(self.labels).type(torch.float32).to(self.device)
        else:
            raise ValueError(f"Unexpected data shape {self.original_ppi.shape}. Expected [n, 2] or [n, 3] shapes.")

        # -----------------------------------------------------------------
        # load sequences into a fixed length code-to-seq dictionary
        # -----------------------------------------------------------------
        aa_dict = AminoAcids.get_amino_acid_symbols_idx_dict()
        self.seq_dict = {code: [aa_dict[aa] for aa in seq]
                         for code, seq in np.loadtxt(seq_filepath, delimiter="\t", dtype=np.str)}
        for code in self.seq_dict:
            if len(self.seq_dict[code]) >= self.max_len:
                seq = torch.from_numpy(np.array(self.seq_dict[code][:max_len])).to(self.device)
            else:
                seq = torch.from_numpy(np.array(self.seq_dict[code] + (self.max_len - len(self.seq_dict[code])) * [0])).to(self.device)
            self.seq_dict[code] = seq

    def __getitem__(self, index):
        """ Get a dataset features and label of record by its index

        Parameters
        ----------
        index: int
            record index

        Returns
        -------
        (torch.Tensor, torch.Tensor)
            record features as two protein index-based sequence tensors (p1_idx_seq, p2_idx_seq)
        torch.Tensor
            record label torch tensor
        """
        p1, p2 = self.p1[index], self.p2[index]
        label = self.labels_tensor[index]
        if p1 != "#":
            return self.encode_ppi_data([p1, p2]), label
        else:
            if self.neg_ppi_source is None:
                p1 = np.random.choice(self._protein_codes, []).tolist()
                p2 = np.random.choice(self._protein_codes, []).tolist()
            else:
                p1, p2 = self.neg_ppi_source[np.random.choice(self.neg_ppi_source.shape[0])].tolist()
            return self.encode_ppi_data([p1, p2]), label

    def __len__(self):
        return self.nb_samples

    def encode_ppi_data(self, ppi_data):
        """ Encode ppi data to pytorch index based sequence features

        Parameters
        ----------
        ppi_data: Iterable
            ppi data Iterable which contain ppis of the shape [n, 2]

        Returns
        -------
        torch.Tensor
            p1 index based sequence tensor
        torch.Tensor
            p2 index based sequence tensor
        """
        if isinstance(ppi_data[0], str):
            return self.seq_dict[ppi_data[0]], self.seq_dict[ppi_data[1]]
        else:
            p1_tensors_list = []
            p2_tensors_list = []
            for p1, p2 in ppi_data:
                p1_tensors_list.append(self.seq_dict[p1])
                p2_tensors_list.append(self.seq_dict[p2])
            p1_tensor = torch.cat(p1_tensors_list, dim=0)
            p2_tensor = torch.cat(p1_tensors_list, dim=0)
            return p1_tensor, p2_tensor

    def add_random_ppis(self, nb_records, ensure_exclusivity=False):
        """ Add extra random ppi records to the dataset

        Parameters
        ----------
        nb_records: int
            Number of ppi records to add
        ensure_exclusivity: bool
            Ensure that the added ppi records are not present in the original dataset instances.
        """
        current_ppis = set([(p1, p2) for p1, p2 in np.concatenate([self.p1.reshape([-1, 1]), self.p2.reshape([-1, 1])], axis=1)])

        random_ppis = np.random.choice(self._protein_codes, [nb_records*2, 2])
        if ensure_exclusivity:
            random_ppis = np.array([[p1, p2] for p1, p2 in random_ppis if (p1, p2) not in current_ppis and (p2, p1) not in current_ppis])

        random_ppis = random_ppis[:nb_records, :]
        new_p1 = np.full([self.p1.shape[0]+random_ppis.shape[0]], "#", dtype=np.dtype('U10'))
        new_p1[:self.p1.shape[0]] = self.p1
        new_p1 = new_p1[self.p1.shape[0]:] = random_ppis[:, 0]
        self.p1 = new_p1

        new_p2 = np.full([self.p2.shape[0] + random_ppis.shape[0]], "#", dtype=np.dtype('U10'))
        new_p2[:self.p2.shape[0]] = self.p2
        new_p2[self.p2.shape[0]:] = random_ppis[:, 0]
        self.p2 = new_p2

        new_y = np.zeros([self.labels.shape[0] + random_ppis.shape[0]], dtype=np.float32)
        new_y[:self.p2.shape[0]] = self.labels
        new_p2[self.labels.shape[0]:] = new_y
        self.labels = new_y
        self.labels_tensor = torch.from_numpy(self.labels).type(torch.float32).to(self.device)

    def get_protein_sequence(self, list_of_proteins):
        """ Get index based sequences for a list of proteins

        Parameters
        ----------
        list_of_proteins: list
            List of protein codes

        Returns
        -------
        torch.tensor
            torch tensor of size (n , l) that represents the proteins' sequences, where n is the number of
            requested protein codes and l is the size of the fixed protein sequence length.
        """
        return torch.cat([self.seq_dict[p].reshape(1, -1) for p in list_of_proteins], dim=0)

    @property
    def protein_codes(self):
        class_counts = np.array([np.sum(self.labels == 0), np.sum(self.labels == 1)])
        class_freq = class_counts/np.max(class_counts)
        return 1 / class_freq


class PpiBalancedBatchSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, balanced_max=None):
        self.dataset_labels = dataset.labels
        self.pos_indices = np.array([idx for idx in range(len(self.dataset_labels)) if self.dataset_labels[idx] == 1])
        self.neg_indices = np.array([idx for idx in range(len(self.dataset_labels)) if self.dataset_labels[idx] == 0])
        np.random.shuffle(self.pos_indices)
        np.random.shuffle(self.neg_indices)

        self.indices = np.arange(len(self.dataset_labels))
        self.balanced_max = balanced_max if balanced_max is not None else len(self.pos_indices)
        self.selected_indices = []

    def __iter__(self):
        selected_indices_pos = self.pos_indices
        selected_indices_neg = np.random.choice(self.neg_indices, self.balanced_max)
        self.selected_indices = np.concatenate([selected_indices_pos, selected_indices_neg])
        np.random.shuffle(self.selected_indices)
        for index in self.selected_indices:
            yield index

    def __len__(self):
        return self.balanced_max * 2
