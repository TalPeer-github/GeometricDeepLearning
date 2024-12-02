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
    # batch_size = 64
    # hidden_channels = 24
    # lr = 0.0025
    # num_epochs = 200
    # wd = 5e-4
    # model_type = "GIN"

    parser = ArgumentParser(description="Graph Classification Task")
    parser.add_argument('--model_type', type=str, help="Chosen Model", default=model_type)
    parser.add_argument('--batch_size', type=int, help="Batch size", default=batch_size)
    parser.add_argument('--hidden_channels', type=int, help="Hidden dimension", default=hidden_channels)
    parser.add_argument('--lr', type=float, help="Learning rate", default=lr)
    parser.add_argument('--wd', type=float, help="Weight decay", default=wd)
    parser.add_argument('--num_epochs', type=int, help="Number of epochs to run", default=num_epochs)

    parser.add_argument('--num_classes', type=int, help="Number of classes", default=2)
    parser.add_argument('--num_node_features', type=int, help="Number of node features", default=7)
    parser.add_argument('--num_edge_attr', type=int, help="Number of edge attributes", default=4)
    return parser.parse_args()


# def initialize_experiments():
#     models = ["SAGE", "GCN", "GAT", "CHEV", "EGC", "GIN"]
#     batch_size = [32, 48, 64]
#     hidden_channels = [16, 24, 36, 48, 64]
#     lr = [0.0005, 0.001, 0.005, 0.01, 0.02, 0.05]
#     num_epochs = [100, 120, 150, 200, 300]
#     wd = [1e-3, 3e-4, 5e-5]
#     results = []
#     for model_type in models:
#         for batch in batch_size:
#             for hidden in hidden_channels:
#                 for learning_rate in lr:
#                     for epochs in num_epochs:
#                         for weight_decay in wd:
#                             args = parse_args()
#                             args.model_type = model_type
#                             args.batch_size = batch
#                             args.hidden_channels = hidden
#                             args.lr = learning_rate
#                             args.num_epochs = epochs
#                             args.wd = weight_decay
#                             config.args = args
#
#                             print(f"Running experiment with {model_type} model")
#                             run_experiment()
#
#                             results.append({
#                                 'model': model_type,
#                                 'batch_size': batch,
#                                 'hidden_channels': hidden,
#                                 'lr': learning_rate,
#                                 'num_epochs': epochs,
#                                 'weight_decay': weight_decay,
#                                 'train_accuracy': 0,
#                                 'val_accuracy': 0,
#                             })
#
#     df_results = pd.DataFrame(results)
#     df_results.to_csv('experiment_results.csv', index=False)
#     print("Results saved to experiment_results.csv")


if __name__ == "__main__":
    exp_models = ["SAGE", "GCN", "GAT", "CHEV", "EGC", "GIN"]
    exp_batch_sizes = [32, 48, 64]
    exp_hidden_channels = [16, 24, 48, 64]
    exp_lrs = [0.0005, 0.001, 0.005, 0.01, 0.025]
    exp_wds = [3e-3, 5e-5]
    exp_epochs = [125, 200]

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

                            print("\n=====================================================")
                            print(f"\nExperiment [{exp_idx} / {num_experiments}] ->"
                                  f"\n\tmodel = {args.model_type}\n\thidden_layers = {args.hidden_channels}"
                                  f"\n\tnum_epochs = {args.num_epochs}\n\tbatch_size = {args.batch_size}"
                                  f"\n\tlr = {args.lr}\n\tweight_decay = {args.wd}\n\n")

                            train_acc, val_acc = run_experiment()
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

                            sleep(1)

    df_results = pd.DataFrame(results)
    df_results.to_csv('../results/experiments_results.csv', index=False)
    print("Experiments results saved to results/experiment_results.csv")

    # args = parse_args()
    # config.args = args
    #
    # from experiments import run_experiment
    #
    # run_experiment()
