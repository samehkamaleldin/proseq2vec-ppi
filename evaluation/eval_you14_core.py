import logging
import warnings
from os.path import join, isdir
from os import makedirs
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, accuracy_score, recall_score
from proseq2vec.data import PpiSeqDataset
from proseq2vec.models import ProSeq2VecPpi
from proseq2vec.util import *
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
import torch

warnings.filterwarnings('ignore')
logging.getLogger("lightning").setLevel(logging.ERROR)


def main():

    seed = 1234
    pl.seed_everything(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tb_log_dirpath = join("../logs", "tb_logs", "you14_core_cv")
    makedirs(tb_log_dirpath) if not isdir(tb_log_dirpath) else None

    # model hparams
    hparams = {
        "lr": 0.00001,
        "seq_len": 2500,
        "aa_k": 8,
        "seq_k": 64,
        "hidden_size": 64,
        "fc_size": 128,
        "nb_layers": 2,
        "batch_size": 256,
        "nb_epochs": 50,
        "dropout": 0.25,
        "activation_type": "none",
    }

    hparams_tag = get_hparams_txt_tag(hparams)
    # Load datasets
    valid_size_ratio = 0.1
    datasets_dp = "../data/you14"
    species_name = "y14_core"
    seq_filepath = join(datasets_dp, "core_seq.txt")
    fold_results = []
    for split_idx in range(5):
        train_valid_pos_ppi_arr = np.loadtxt(join(datasets_dp, f"cv_splits/split_{split_idx}", f"train_pos.txt"), dtype=np.dtype('U20'))
        train_valid_neg_ppi_arr = np.loadtxt(join(datasets_dp, f"cv_splits/split_{split_idx}", f"train_neg.txt"), dtype=np.dtype('U20'))
        train_valid_pos_ppi_arr_labels = np.ones([train_valid_pos_ppi_arr.shape[0], 1])
        train_valid_neg_ppi_arr_labels = np.zeros([train_valid_neg_ppi_arr.shape[0], 1])
        train_valid_pos_ppi_arr = np.concatenate([train_valid_pos_ppi_arr, train_valid_pos_ppi_arr_labels], axis=1)
        train_valid_neg_ppi_arr = np.concatenate([train_valid_neg_ppi_arr, train_valid_neg_ppi_arr_labels], axis=1)
        train_pos_ppi_arr, valid_pos_ppi_arr = train_test_split(train_valid_pos_ppi_arr, test_size=valid_size_ratio)
        train_neg_ppi_arr, valid_neg_ppi_arr = train_test_split(train_valid_neg_ppi_arr, test_size=valid_size_ratio)
        train_data_arr = np.concatenate([train_pos_ppi_arr, train_neg_ppi_arr], axis=0)
        valid_data_arr = np.concatenate([valid_pos_ppi_arr, valid_neg_ppi_arr], axis=0)

        test_pos_ppi_arr = np.loadtxt(join(datasets_dp, f"cv_splits/split_{split_idx}", "test_pos.txt"), dtype=np.dtype('U20'))
        test_neg_ppi_arr = np.loadtxt(join(datasets_dp, f"cv_splits/split_{split_idx}", "test_neg.txt"), dtype=np.dtype('U20'))
        test_pos_ppi_arr_labels = np.ones([test_pos_ppi_arr.shape[0], 1])
        test_neg_ppi_arr_labels = np.zeros([test_neg_ppi_arr.shape[0], 1])
        test_pos_ppi_arr = np.concatenate([test_pos_ppi_arr, test_pos_ppi_arr_labels], axis=1)
        test_neg_ppi_arr = np.concatenate([test_neg_ppi_arr, test_neg_ppi_arr_labels], axis=1)
        test_data_arr = np.concatenate([test_pos_ppi_arr, test_neg_ppi_arr], axis=0)

        print(f"= Loading training/validation/testing data (fold: {split_idx}) ...")
        train_dataset = PpiSeqDataset(train_data_arr, seq_filepath, max_len=hparams["seq_len"], device=device)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=hparams["batch_size"], shuffle=True)

        valid_dataset = PpiSeqDataset(valid_data_arr, seq_filepath, max_len=hparams["seq_len"], device=device)
        valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=hparams["batch_size"], shuffle=True)

        test_all_dataset = PpiSeqDataset(test_data_arr, seq_filepath, max_len=hparams["seq_len"], device=device)
        test_dataloader = DataLoader(dataset=test_all_dataset, batch_size=hparams["batch_size"], shuffle=False)

        # Model training
        p2v_model = ProSeq2VecPpi(**hparams).to(device)
        tb_logger = TensorBoardLogger(tb_log_dirpath, name=f'basic_p2v_sp={species_name}__{hparams_tag}', log_graph=True, default_hp_metric=False)
        pl_trainer = pl.Trainer(reload_dataloaders_every_epoch=True, max_epochs=hparams["nb_epochs"], check_val_every_n_epoch=3, logger=tb_logger, checkpoint_callback=False, weights_summary=None, progress_bar_refresh_rate=0)
        pl_trainer.fit(p2v_model, train_dataloader=train_dataloader, val_dataloaders=valid_dataloader)

        test_test_dataloaders_list = test_dataloader
        test_results = pl_trainer.test(p2v_model, test_test_dataloaders_list, verbose=False)[0]

        test_logits = test_results['logits']
        test_labels = test_results['labels']
        test_decisions = np.array(test_logits[:, 1] > test_logits[:, 0], dtype=np.int)
        pre = precision_score(test_labels, test_decisions)
        acc = accuracy_score(test_labels, test_decisions)
        rec = recall_score(test_labels, test_decisions)
        fold_results.append([pre, rec, acc])
        print("+---------------------+------------------+-----------------+-----------------+")
        print(f"| you14-core (fold:{split_idx}) | precision: {pre:0.3f} | recall: {rec:0.3f}   | accuracy: {acc:0.3f} |")
        print("+---------------------+------------------+-----------------+-----------------+")
    fold_results_avg = np.mean(np.array(fold_results), axis=0)
    avg_pre, avg_rec, avg_acc = fold_results_avg[0], fold_results_avg[1], fold_results_avg[2]
    print("+---------------------+-------------------+------------------+------------------+")
    print(f"| you14-core  (avg.)  | precision: {avg_pre:0.4f} | recall: {avg_rec:0.4f}   | accuracy: {avg_acc:0.4f} |")
    print("+---------------------+-------------------+------------------+------------------+")


if __name__ == '__main__':
    main()
