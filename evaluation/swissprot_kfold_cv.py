import warnings
import argparse
from os.path import join, isdir, isfile
from os import makedirs

from sklearn.model_selection import train_test_split, KFold

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from ax.service.ax_client import AxClient
from ax.modelbridge.generation_strategy import GenerationStrategy, GenerationStep
from ax import save as save_ax_axp, load as load_ax_exp, Models
from proseq2vec.data import PpiSeqDataset
from proseq2vec.models import ProSeq2VecPpi
from proseq2vec.util import *

warnings.filterwarnings('ignore')


def main():
    parser = argparse.ArgumentParser(description='ProSeq2Vec PPI prediction model cross validation evaluation script')
    parser.add_argument('-p', '--species', type=str, help='PPI species', required=False, default="ecoli")
    parser.add_argument('-s', '--seq_len', type=int, help='Max allowed protein sequence length', required=False, default=2500)
    args = vars(parser.parse_args())

    # ------------------------------------------------------------
    # Paths and directories
    datasets_dp = "../data/"
    logs_dp = "../logs"
    species_name = args['species']
    ax_logs_dp = join(logs_dp, "ax_logs", "cv", "swissprot", f"{species_name}")
    makedirs(ax_logs_dp) if not isdir(ax_logs_dp) else None

    # ------------------------------------------------------------
    # Model searchable and fixed hyperparameters
    ax_hparams = [
        {"name": "aa_k", "type": "range", "value_type": "int", "bounds": [8, 24]},
        {"name": "seq_k", "type": "range", "value_type": "int", "bounds": [128, 512]},
        {"name": "hidden_size", "type": "range", "value_type": "int", "bounds": [128, 512]},
        {"name": "dropout", "type": "range", "value_type": "float", "bounds": [0.01, 0.4]},
    ]

    # model fixed_params
    fixed_params = {
        "lr": 0.001,
        "seq_len": 2500,
        "nb_layers": 2,
        "fc_size": 1024,
        "activation_type": "leaky_relu",
        "batch_size": 128,
        "nb_epochs": 200,
    }

    best_param_nb_epochs = 400
    # ------------------------------------------------------------
    # experiment configuration
    seed = 1234
    pl.seed_everything(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nb_k_splits = 5
    nb_runs = 3
    nb_ax_sobol_trials = len(ax_hparams) + 1
    nb_ax_all_trials = 30
    valid_size_ratio = 0.2

    # ------------------------------------------------------------
    # Load datasets
    seq_filepath = join(datasets_dp, "swissprot_ppi", "seq.txt")
    pos_data_arr = np.loadtxt(join(datasets_dp, "swissprot_ppi", f"ppi_{species_name}_pos.txt"), dtype=np.dtype('U20'))
    neg_data_arr = np.loadtxt(join(datasets_dp, "swissprot_ppi", f"ppi_{species_name}_neg.txt"), dtype=np.dtype('U20'))

    kfold_module = KFold(n_splits=nb_k_splits, shuffle=True)
    run_metrics_list = []
    for run_idx in range(nb_runs):
        pos_splits = list(kfold_module.split(pos_data_arr))
        neg_splits = list(kfold_module.split(neg_data_arr))
        fold_metrics_list = []
        for fold_idx, (train_pos_indices, test_pos_indices) in enumerate(pos_splits):
            train_valid_neg_indices, test_neg_indices = neg_splits[fold_idx]
            train_valid_neg_arr = neg_data_arr[train_valid_neg_indices, :]
            train_valid_pos_ppi_arr = pos_data_arr[train_pos_indices, :]
            train_pos_ppi_arr, valid_pos_ppi_arr = train_test_split(train_valid_pos_ppi_arr, test_size=valid_size_ratio)
            _, valid_ppi_neg_arr = train_test_split(train_valid_neg_arr, test_size=valid_size_ratio)

            valid_data_arr = np.concatenate([valid_pos_ppi_arr, valid_ppi_neg_arr])
            test_data_arr = np.concatenate([pos_data_arr[test_pos_indices, :], neg_data_arr[test_neg_indices, :]])

            train_dataset = PpiSeqDataset(train_pos_ppi_arr, seq_filepath, neg2pos=1, max_len=fixed_params["seq_len"], device=device)
            train_dataloader = DataLoader(dataset=train_dataset, batch_size=fixed_params["batch_size"], shuffle=True)
            valid_dataset = PpiSeqDataset(valid_data_arr, seq_filepath, max_len=fixed_params["seq_len"], device=device)
            valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=fixed_params["batch_size"], shuffle=True)
            test_all_dataset = PpiSeqDataset(test_data_arr, seq_filepath, max_len=fixed_params["seq_len"], device=device)
            test_dataloader = DataLoader(dataset=test_all_dataset, batch_size=fixed_params["batch_size"], shuffle=False)

            # ------------------------------------------------------------
            # Configure and run Ax optimization
            # ------------------------------------------------------------
            def eval_ax_hparams(hparams, trial_idx=-1):
                hparams_tag = get_hparams_txt_tag(fixed_params) + "__" + get_hparams_txt_tag(hparams)
                ax_p2v_model = ProSeq2VecPpi(**fixed_params, **hparams, tb_log_mode="info").to(device)
                ax_pl_trainer = pl.Trainer(reload_dataloaders_every_epoch=True, max_epochs=fixed_params["nb_epochs"], check_val_every_n_epoch=10, progress_bar_refresh_rate=0, weights_summary=None, checkpoint_callback=False)
                ax_pl_trainer.fit(ax_p2v_model, train_dataloader=train_dataloader, val_dataloaders=valid_dataloader)
                ax_valid_results = ax_pl_trainer.test(ax_p2v_model, valid_dataloader, verbose=False)
                return ax_valid_results[0]["metrics"]
            # ------------------------------------------------------------
            gs = GenerationStrategy(steps=[
                GenerationStep(Models.SOBOL, num_trials=nb_ax_sobol_trials),
                GenerationStep(Models.GPEI, num_trials=-1)])
            # ------------------------------------------------------------
            ax_exp_name = f"cv_p2v_sp=swissprot_{species_name}_run={run_idx}_fold={fold_idx}_ax"
            ax_client_filepath = join(ax_logs_dp, f"{ax_exp_name}_client.json")
            ax_exp_filepath = join(ax_logs_dp, f"{ax_exp_name}_experiment.json")
            ax_client = AxClient(generation_strategy=gs)
            if isfile(ax_exp_filepath) and isfile(ax_client_filepath):
                ax_client.load_from_json_file(ax_client_filepath)
                ax_client._experiment = load_ax_exp(ax_exp_filepath)
            else:
                ax_client.create_experiment(name=ax_exp_name, parameters=ax_hparams, objective_name="ap")

            while ax_client.experiment.num_trials < nb_ax_all_trials:
                parameters, trial_index = ax_client.get_next_trial()
                results = eval_ax_hparams(parameters, trial_index)
                ax_client.complete_trial(trial_index=trial_index, raw_data=results["ap"])
                # storing ax model and experiment
                ax_client.save_to_json_file(ax_client_filepath)
                save_ax_axp(ax_client.experiment, ax_exp_filepath)

            best_params, values = ax_client.get_best_parameters()
            # ------------------------------------------------------------
            # train and eval on testing data
            best_hparams_tag = get_hparams_txt_tag(fixed_params) + "__" + get_hparams_txt_tag(best_params)
            p2v_model = ProSeq2VecPpi(**fixed_params, **best_params).to(device)
            pl_trainer = pl.Trainer(reload_dataloaders_every_epoch=True, max_epochs=best_param_nb_epochs, check_val_every_n_epoch=5, progress_bar_refresh_rate=0, weights_summary=None, checkpoint_callback=False)
            pl_trainer.fit(p2v_model, train_dataloader=train_dataloader, val_dataloaders=valid_dataloader)
            test_results = pl_trainer.test(p2v_model, test_dataloader, verbose=False)
            print_model_results(f"run:{run_idx:02d}-fold:{fold_idx:02d}", test_results[0])
            fold_mt = test_results[0]["metrics"]
            print(f"#=[sp:{species_name}]=> run:{run_idx:02d} - fold:{fold_idx:02d} - ap:{fold_mt['ap']:0.4f}  => fold_metrics:{fold_mt} - best_hparams:{best_params}_{fixed_params}", flush=True)
            fold_metrics_list.append(fold_mt)

        run_ap = np.mean([r["ap"] for r in fold_metrics_list])
        run_avg_metrics = {k: np.mean([m[k] for m in fold_metrics_list]) for k in fold_metrics_list[0].keys()}
        run_metrics_list.append(run_avg_metrics)
        print(f"#=[sp:{species_name}]=> run:{run_idx:02d} - fold:** - ap:{run_ap:0.4f}  => [run-average] metrics: {run_avg_metrics}", flush=True)

    mean_acc = np.mean([r["accuracy"] for r in run_metrics_list])
    mean_ap = np.mean([r["ap"] for r in run_metrics_list])
    print("\n+------------+--------------+-----------------+")
    print(f"| Cross Validation - Runs Average             |")
    print("+------------+--------------+-----------------+")
    print("| Model      |   Accuracy   | Avg. Precision  |")
    print("+------------+--------------+-----------------+")
    print(f"| proseq2vec |    {mean_acc:0.4f}    |     {mean_ap:0.4f}      |")
    print("+------------+--------------+-----------------+")


if __name__ == '__main__':
    main()
