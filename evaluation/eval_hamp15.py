import argparse
import logging
from os.path import join, isdir, isfile
from os import makedirs
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, auc


from proseq2vec.data import PpiSeqDataset
from proseq2vec.models import ProSeq2VecPpi
from proseq2vec.util import *

from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
import torch

import warnings

warnings.filterwarnings('ignore')
logging.getLogger("lightning").setLevel(logging.ERROR)


def get_dataloaders(datasets_dp, seq_filepath, run_idx, fold_idx, valid_size_ratio, batch_size, seq_len):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_valid_pos_ppi_arr = np.loadtxt(join(datasets_dp, f"split_{run_idx}", f"{fold_idx}.train.pos"), dtype=np.dtype('U20'))[:, :2]
    train_valid_neg_ppi_arr = np.loadtxt(join(datasets_dp, f"split_{run_idx}", f"{fold_idx}.train.neg"), dtype=np.dtype('U20'))
    train_valid_pos_ppi_arr_labels = np.ones([train_valid_pos_ppi_arr.shape[0], 1])
    train_valid_neg_ppi_arr_labels = np.zeros([train_valid_neg_ppi_arr.shape[0], 1])
    train_valid_pos_ppi_arr = np.concatenate([train_valid_pos_ppi_arr, train_valid_pos_ppi_arr_labels], axis=1)
    train_valid_neg_ppi_arr = np.concatenate([train_valid_neg_ppi_arr, train_valid_neg_ppi_arr_labels], axis=1)
    train_pos_ppi_arr, valid_pos_ppi_arr = train_test_split(train_valid_pos_ppi_arr, test_size=valid_size_ratio)
    train_neg_ppi_arr, valid_neg_ppi_arr = train_test_split(train_valid_neg_ppi_arr, test_size=valid_size_ratio)

    train_data_arr = np.concatenate([train_pos_ppi_arr, train_neg_ppi_arr], axis=0)
    valid_data_arr = np.concatenate([valid_pos_ppi_arr, valid_neg_ppi_arr], axis=0)

    test_pos_ppi_arr = np.loadtxt(join(datasets_dp, f"split_{run_idx}", f"{fold_idx}.test.pos"), dtype=np.dtype('U20'))[:, :2]
    test_neg_ppi_arr = np.loadtxt(join(datasets_dp, f"split_{run_idx}", f"{fold_idx}.test.neg"), dtype=np.dtype('U20'))
    test_pos_ppi_arr_labels = np.ones([test_pos_ppi_arr.shape[0], 1])
    test_neg_ppi_arr_labels = np.zeros([test_neg_ppi_arr.shape[0], 1])
    test_pos_ppi_arr = np.concatenate([test_pos_ppi_arr, test_pos_ppi_arr_labels], axis=1)
    test_neg_ppi_arr = np.concatenate([test_neg_ppi_arr, test_neg_ppi_arr_labels], axis=1)
    test_data_arr = np.concatenate([test_pos_ppi_arr, test_neg_ppi_arr], axis=0)

    train_dataset = PpiSeqDataset(train_pos_ppi_arr, seq_filepath, neg2pos=1, max_len=seq_len, negatives_source=train_neg_ppi_arr, device=device)
    valid_dataset = PpiSeqDataset(valid_data_arr, seq_filepath, max_len=seq_len, device=device)
    test_all_dataset = PpiSeqDataset(test_data_arr, seq_filepath, max_len=seq_len, device=device)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=test_all_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader, valid_dataloader, test_dataloader


def eval_hamp15_config(species_name, run_index, fold_index):
    seed = 1234
    pl.seed_everything(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tb_logs_dp = join("../logs", "tb_logs", "eval", "hamp15", species_name)
    makedirs(tb_logs_dp) if not isdir(tb_logs_dp) else None

    # model fixed_params
    hparams = {
        "lr": 0.000005,
        "aa_k": 16,
        "seq_k": 256,
        "dropout": 0.15,
        "hidden_size": 256,
        "nb_layers": 5,
        "fc_size": 512,
        "activation_type": "none",
        "batch_size": 1024,
        "nb_epochs": 300,
        "seq_len": 4000,
    }

    hparams_tag = get_hparams_txt_tag(hparams)
    # Load datasets
    valid_size_ratio = 0.2
    datasets_dp = f"../data/hamp15/{species_name}CV"
    seq_filepath = join(datasets_dp, f"{species_name}.seq")
    train_dataloader, valid_dataloader, test_dataloader = get_dataloaders(datasets_dp, seq_filepath, run_index, fold_index, valid_size_ratio, hparams['batch_size'], hparams['seq_len'])

    p2v_model = ProSeq2VecPpi(**hparams).to(device)
    tb_logger = TensorBoardLogger(tb_logs_dp, name=f'proseq2vec_hamp15_sp={species_name}_run={run_index}_fold={fold_index}__{hparams_tag}', log_graph=False, default_hp_metric=False)
    pl_trainer = pl.Trainer(reload_dataloaders_every_epoch=True, max_epochs=hparams["nb_epochs"], check_val_every_n_epoch=1, logger=tb_logger, checkpoint_callback=False, weights_summary=None, progress_bar_refresh_rate=0)

    pl_trainer.fit(p2v_model, train_dataloader=train_dataloader, val_dataloaders=valid_dataloader)
    test_results = pl_trainer.test(p2v_model, test_dataloader, verbose=False)[0]

    test_logits = test_results['logits']
    test_labels = test_results['labels']
    test_scores = test_logits[:, 1]
    pre, rec, thr = precision_recall_curve(test_labels, test_scores)
    auc_pr = auc(rec, pre)
    fold_auc_pr = auc_pr
    print(f"| hamp15-{species_name} (run:{run_index}-fold:{fold_index}) | auc-pr: {fold_auc_pr:0.4f} |")


def main():
    parser = argparse.ArgumentParser(description='ProSeq2Vec PPI prediction on Hamp15 dataset')
    parser.add_argument('-s', '--species', type=str, help='PPI species', required=False, default="human")
    parser.add_argument('-r', '--run', type=int, help='Hamp15 run index', required=False, default=0)
    parser.add_argument('-f', '--fold', type=int, help='Hamp15 split index', required=False, default=0)
    args = vars(parser.parse_args())

    species = args["species"]
    run = args["run"]
    fold = args["fold"]

    eval_hamp15_config(species, run, fold)


if __name__ == '__main__':
    main()
