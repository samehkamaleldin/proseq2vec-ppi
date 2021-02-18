# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from tabulate import tabulate


class AminoAcids:
    amino_acid_dict = {
        "alanine": "A",
        "aspartic/asparagine": "B",
        "arginine": "R",
        "asparagine": "N",
        "aspartic": "D",
        "cysteine": "C",
        "glutamine": "Q",
        "glutamic": "E",
        "glycine": "G",
        "histidine": "H",
        "isoleucine": "I",
        "leucine": "L",
        "lysine": "K",
        "methionine": "M",
        "phenylalanine": "F",
        "pyrrolysine": "O",
        "proline": "P",
        "serine": "S",
        "selenocysteine": "U",
        "threonine": "T",
        "tryptophan": "W",
        "unknown": "X",
        "tyrosine": "Y",
        "valine": "V",
        "glutamic/glutamine": "Z"
    }

    @staticmethod
    def get_amino_acids_count():
        return len(AminoAcids.get_amino_acid_symbols())

    @staticmethod
    def get_amino_acid_symbols():
        return list(AminoAcids.amino_acid_dict.values())

    @staticmethod
    def get_amino_acid_symbols_idx_dict():
        symbols = sorted(AminoAcids.get_amino_acid_symbols())
        return {symbols[v]: v+1 for v in range(len(symbols))}

    @staticmethod
    def get_amino_acid_names():
        return list(AminoAcids.amino_acid_dict.keys())


def hits_at_k_score(y_true: np.ndarray, y_score: np.ndarray, k: int):
    """ Compute hits at k metric score

    Parameters
    ----------
    y_true: np.ndarray
        True labels
    y_score: np.ndarray
        Predicted scores
    k: int
        the k position

    Returns
    -------
    float
        The hits@k score
    """
    if k > y_true.shape[0]:
        raise ValueError("Value of k must be less the the size of provided true labels and scores.")
    if y_true.shape[0] != y_score.shape[0]:
        raise ValueError(f"size mismatch between y_true of shape {y_true.shape} and y_score of shape {y_score.shape}.")
    sorted_indices = np.argsort(y_score)[::-1]
    sorted_labels = y_true[sorted_indices]
    return np.sum(sorted_labels[:k])


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int):
    """ Compute precision at k metric score

    Parameters
    ----------
    y_true: np.ndarray
        True labels with positives as ones and negatives as zeros
    y_score: np.ndarray
        Predicted scores
    k: int
        the k position

    Returns
    -------
    float
        The precision@k score
    """
    if k > y_true.shape[0]:
        raise ValueError("Value of k must be less the the size of provided true labels and scores.")
    if y_true.shape[0] != y_score.shape[0]:
        raise ValueError(f"size mismatch between y_true of shape {y_true.shape} and y_score of shape {y_score.shape}.")
    nb_pos = np.sum(y_true)
    hits_at_k = hits_at_k_score(y_true, y_score, k)
    return hits_at_k / np.min([nb_pos, k])


def print_model_results(model_name, metrics_dict, floatfmt=".2f"):
    cols = ['model'] + sorted(metrics_dict.keys())
    row = {"model": model_name}
    row.update(metrics_dict)
    results_df = pd.DataFrame(columns=cols)
    results_df = results_df.append(row, ignore_index=True)
    print(tabulate(results_df, headers='keys', tablefmt="psql", floatfmt=floatfmt, showindex=False))


def get_hparams_txt_tag(hparams: dict, params_set: set = None):
    txt_list = []
    for k, v in hparams.items():
        if params_set is None:
            p_txt = f"{k}={v}" if not isinstance(v, float) else f"{k}={v:0.5f}"
            txt_list.append(p_txt)
        elif k in params_set:
            p_txt = f"{k}={v}" if not isinstance(v, float) else f"{k}={v:0.5f}"
            txt_list.append(p_txt)
    return "_".join(txt_list)
