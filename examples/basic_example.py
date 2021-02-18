
from os.path import join, isdir
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from proseq2vec.data import PpiSeqDataset
from proseq2vec.models import ProSeq2VecPpi
from proseq2vec.util import *
import pytorch_lightning as pl
import torch

import warnings
warnings.filterwarnings('ignore')


def main():

    seed = 1234
    pl.seed_everything(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model hparams
    hparams = {
        "lr": 0.001,
        "seq_len": 2500,
        "aa_k": 10,
        "seq_k": 128,
        "hidden_size": 256,
        "fc_size": 512,
        "nb_layers": 2,
        "batch_size": 128,
        "nb_epochs": 1,
        "dropout": 0.35,
        "activation_type": "leaky_relu",
    }

    hparams_tag = get_hparams_txt_tag(hparams)
    # Load datasets
    valid_size_ratio = 0.2
    datasets_dp = "../data/"
    species_name = "ecoli"
    seq_filepath = join(datasets_dp, "swissprot_ppi", "seq.txt")
    pos_data_arr = np.loadtxt(join(datasets_dp, "swissprot_ppi", f"ppi_{species_name}_pos.txt"), dtype=np.dtype('U20'))[:100]
    neg_data_arr = np.loadtxt(join(datasets_dp, "swissprot_ppi", f"ppi_{species_name}_neg.txt"), dtype=np.dtype('U20'))[:100]

    train_valid_pos_ppi_arr, test_pos_ppi_arr = train_test_split(pos_data_arr, test_size=valid_size_ratio)
    train_valid_neg_ppi_arr, test_neg_ppi_arr = train_test_split(neg_data_arr, test_size=valid_size_ratio)
    train_pos_ppi_arr, valid_pos_ppi_arr = train_test_split(train_valid_pos_ppi_arr, test_size=valid_size_ratio)
    _, valid_neg_ppi_arr = train_test_split(train_valid_neg_ppi_arr, test_size=valid_size_ratio)
    valid_data_arr = np.concatenate([valid_pos_ppi_arr, valid_neg_ppi_arr], axis=0)
    test_data_arr = np.concatenate([test_pos_ppi_arr, test_neg_ppi_arr], axis=0)

    print("= Loading training/validation/testing data ...")
    train_pos_dataset = PpiSeqDataset(train_pos_ppi_arr, seq_filepath, neg2pos=1, max_len=hparams["seq_len"], device=device)
    train_dataloader = DataLoader(dataset=train_pos_dataset, batch_size=hparams["batch_size"], shuffle=True)
    valid_pos_dataset = PpiSeqDataset(valid_data_arr, seq_filepath, max_len=hparams["seq_len"], device=device)
    valid_dataloader = DataLoader(dataset=valid_pos_dataset, batch_size=hparams["batch_size"], shuffle=True)
    test_all_dataset = PpiSeqDataset(test_data_arr, seq_filepath, max_len=hparams["seq_len"], device=device)
    test_dataloader = DataLoader(dataset=test_all_dataset, batch_size=hparams["batch_size"], shuffle=False)

    # Model training
    p2v_model = ProSeq2VecPpi(**hparams).to(device)
    pl_trainer = pl.Trainer(reload_dataloaders_every_epoch=True, max_epochs=hparams["nb_epochs"], check_val_every_n_epoch=5, checkpoint_callback=False)
    pl_trainer.fit(p2v_model, train_dataloader=train_dataloader, val_dataloaders=valid_dataloader)

    print("=> Log test dataset protein sequence embeddings")
    test_pc = test_all_dataset.protein_codes
    test_seqs = test_all_dataset.get_protein_sequence(test_pc)
    seq_em = p2v_model.get_seq_embeddings(test_seqs, True, test_pc)

    test_test_dataloaders_list = test_dataloader
    test_results = pl_trainer.test(p2v_model, test_test_dataloaders_list, verbose=False)[0]
    test_metrics = test_results['metrics']
    test_ap_txt = f"{test_metrics['ap']:0.4f}"
    test_acc_txt = f"{test_metrics['accuracy']:0.4f}"
    print("+------------+--------------+-----------------+")
    print("| Model      |   Accuracy   | Avg. Precision  |")
    print("+------------+--------------+-----------------+")
    print(f"| proseq2vec |    {test_acc_txt}    |     {test_ap_txt}      |")
    print("+------------+--------------+-----------------+")


if __name__ == '__main__':
    main()
