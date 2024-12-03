import numpy as np

import config
from argparse import ArgumentParser
import pandas as pd
from datasets_class.custom_graph_dataset import *
from time import sleep


def parse_args(num_epochs, batch_size, lr, wd, model_type, hidden_channels):
    """
    # ChevNet works will with LR 0.001 and num_epochs 120 weight_decay 5e-5 batch_size = 64 hidden_channels=24
    # EGC works will with LR 0.001 and num_epochs 120 weight_decay 5e-5 batch_size = 64 hidden_channels=24
    # SAGE works well with LR 0.01 and 125 Epochs batch_size = 32 hidden_channels=32
    # GAT works well with LR 0.0005 and 100 Epochs batch_size = 32 hidden_channels=32 # small LR in general
    :return:
    """

    parser = ArgumentParser(description="Graph Classification Task")
    parser.add_argument('--model_type', type=str, help="Chosen Model", default=model_type)
    parser.add_argument('--k_order', type=int, help="Polynomial order for ChevConv", default=k_order)
    parser.add_argument('--hidden_channels', type=int, help="Hidden dimension", default=hidden_channels)

    parser.add_argument('--batch_size', type=int, help="Batch size", default=batch_size)
    parser.add_argument('--lr', type=float, help="Learning rate", default=lr)
    parser.add_argument('--wd', type=float, help="Weight decay", default=wd)
    parser.add_argument('--num_epochs', type=int, help="Number of epochs to run", default=num_epochs)

    parser.add_argument('--num_classes', type=int, help="Number of classes", default=2)
    parser.add_argument('--num_node_features', type=int, help="Number of node features", default=7)
    parser.add_argument('--num_edge_attr', type=int, help="Number of edge attributes", default=4)
    return parser.parse_args()


if __name__ == "__main__":
    exp_models = ["SAGE", "GCN", "GAT", "ChebNet", "EGC", "GIN"]
    exp_batch_sizes = [32, 48, 64]
    exp_hidden_channels = [24, 48, 64]
    exp_lrs = [0.0005, 0.001, 0.005, 0.01]
    exp_lrs = [5e-4, 1e-3, 5e-3, 1e-2]
    exp_wds = [3e-3, 5e-5]
    exp_epochs = [125, 200]
    k_ords = [2, 3, 4]
    num_lins = [1, 2]
    num_convs = [2, 3, 4, 5]
    poolings = ["add", "mean", "max"]
    results = []
    demo_run = -1

    num_experiments = len(exp_models) * len(exp_batch_sizes) * len(exp_hidden_channels) * len(exp_lrs) * \
                      len(exp_wds) * len(exp_epochs)
    exp_idx = 0

    for epochs in exp_epochs[:demo_run]:
        for batch in exp_batch_sizes[:demo_run]:
            for learning_rate in exp_lrs[:demo_run]:
                for weight_decay in exp_wds[:demo_run]:
                    for model_type in exp_models[:demo_run]:
                        for hidden in exp_hidden_channels[:demo_run]:

                            args = parse_args(epochs, batch, learning_rate, weight_decay, model_type, hidden)
                            args.model_type = model_type
                            args.batch_size = batch
                            args.hidden_channels = hidden
                            args.lr = learning_rate
                            args.num_epochs = epochs
                            args.wd = weight_decay
                            config.args = args

                            train_acc, val_acc = [], []

                            from experiments import run_experiment

                            exp_idx += 1
                            if exp_idx % 50 == 0 or exp_idx == 1:
                                print("\n=====================================================")
                                print(f"\nExperiment [{exp_idx} / {num_experiments}] ->"
                                      f"\n\tmodel = {args.model_type}\n\thidden_layers = {args.hidden_channels}"
                                      f"\n\tnum_epochs = {args.num_epochs}\n\tbatch_size = {args.batch_size}"
                                      f"\n\tlr = {args.lr}\n\tweight_decay = {args.wd}\n\n")

                            train_loss, val_loss, train_acc, val_acc = run_experiment()
                            train_max_acc, train_max_acc_epoch, train_mean_acc, train_acc_var = np.max(
                                train_acc), np.argmax(train_acc), np.mean(train_acc), np.var(train_acc)
                            val_max_acc, val_max_acc_epoch, val_mean_acc, val_acc_var = np.max(val_acc), np.argmax(
                                val_acc), np.mean(val_acc), np.var(val_acc)

                            results.append({
                                'model': model_type,
                                'num_epochs': epochs,
                                'batch_size': batch,
                                'hidden_channels': hidden,
                                'lr': learning_rate,
                                'weight_decay': weight_decay,
                                'mean_train_accuracy': train_mean_acc,
                                'var_train_accuracy': train_acc_var,
                                'max_train_accuracy': train_max_acc,
                                'max_train_accuracy_epoch': train_max_acc_epoch,
                                'mean_val_accuracy': val_mean_acc,
                                'var_val_accuracy': val_acc_var,
                                'max_val_accuracy': val_max_acc,
                                'max_val_accuracy_epoch': val_max_acc_epoch,
                            })
                            if exp_idx % 50 == 0 or exp_idx == 1:
                                print(
                                    f"-> Train max accuracy = {train_max_acc:.4f} | Validation max accuracy = {val_max_acc:.4f}")

                            sleep(1)

    df_results = pd.DataFrame(results)
    df_results.to_csv('../results/experiments_results2.csv', index=False)
    print("Experiments results saved to results/experiment_results2.csv")
