import argparse
import logging
import warnings
from os.path import join
from torch.utils.data import DataLoader
import numpy as np
from p2v.data import PpiSeqDataset
from p2v.models import ProSeq2Vec
import pytorch_lightning as pl
import torch
from datetime import datetime


warnings.filterwarnings('ignore')
logging.getLogger('lightning').setLevel(0)


def main():
    parser = argparse.ArgumentParser(description='ProSeq2Vec PPI prediction model cross validation evaluation script')
    parser.add_argument('-s', '--seq'       , type=int, help='Max allowed protein sequence length'    , required=False, default=1000)
    parser.add_argument('-d', '--data_size' , type=int, help='Data size, or th number of ppi records' , required=False, default=1000)
    parser.add_argument('-l', '--nb_layers' , type=int, help='Number of layers' , required=False, default=2)
    parser.add_argument('-k', '--seq_k'     , type=int, help='Sequence embedding size' , required=False, default=64)
    parser.add_argument('-a', '--aa_k'      , type=int, help='Amino acid embedding size' , required=False, default=8)
    parser.add_argument('-n', '--hidden'    , type=int, help='LSTM hidden size' , required=False, default=128)
    parser.add_argument('-f', '--fc_size'   , type=int, help='LSTM fully connected size' , required=False, default=128)
    parser.add_argument('-t', '--tag'       , type=str, help='experiment_tag' , required=False, default="exp")

    args = vars(parser.parse_args())
    seed = 1234
    pl.seed_everything(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model hparams
    hparams = {
        "seq_len": args['seq'],
        "em_size_aa": args['aa_k'],
        "em_size_seq": args['seq_k'],
        "hidden_size": args['hidden'],
        "fc_size": args['fc_size'],
        "nb_layers": args['nb_layers'],
        "batch_size": 512,
        "nb_epochs": 10,
        "dropout": 0.25,
        "lr": 0.001,
    }

    nb_ppi = args['data_size']
    # Load datasets
    datasets_dp = "../data/datasets/"
    species_name = "human"
    seq_filepath = join(datasets_dp, "swissprot_ppi", "seq.txt")
    pos_data_arr = np.loadtxt(join(datasets_dp, "swissprot_ppi", f"ppi_{species_name}_pos.txt"), dtype=np.dtype('U20'))[:nb_ppi]

    train_pos_dataset = PpiSeqDataset(pos_data_arr, seq_filepath, neg2pos=1, max_len=hparams["seq_len"], device=device)
    train_dataloader = DataLoader(dataset=train_pos_dataset, batch_size=hparams["batch_size"], shuffle=True)

    # Model training
    p2v_model = ProSeq2Vec(**hparams).to(device)
    pl_trainer = pl.Trainer(reload_dataloaders_every_epoch=True, max_epochs=hparams["nb_epochs"], logger=False, log_every_n_steps=1000, checkpoint_callback=False, progress_bar_refresh_rate=0, num_sanity_val_steps=0)
    start_time = datetime.now()
    pl_trainer.fit(p2v_model, train_dataloader=train_dataloader)
    end_time = datetime.now()
    train_time = end_time - start_time
    train_time_sec = train_time.total_seconds()
    print(f"{args['tag']}, {train_time_sec:1.1f}", flush=True)


if __name__ == '__main__':
    main()
