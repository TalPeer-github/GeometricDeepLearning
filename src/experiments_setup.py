import numpy as np
import itertools
import config
from argparse import ArgumentParser
import pandas as pd
from datasets_class.custom_graph_dataset import *
from time import sleep


def parse_args(exp_params):
    """
    # ChevNet works will with LR 0.001 and num_epochs 120 weight_decay 5e-5 batch_size = 64 hidden_channels=24
    # EGC works will with LR 0.001 and num_epochs 120 weight_decay 5e-5 batch_size = 64 hidden_channels=24
    # SAGE works well with LR 0.01 and 125 Epochs batch_size = 32 hidden_channels=32
    # GAT works well with LR 0.0005 and 100 Epochs batch_size = 32 hidden_channels=32 # small LR in general
    :return:
    """

    model_type, batch_size, hidden_channels, lr, wd, num_epochs, k_order, n_linear, n_conv, pool = exp_params
    parser = ArgumentParser(description="Graph Classification Task")
    parser.add_argument('--model_type', type=str, help="Chosen Model", default=model_type)
    parser.add_argument('--k_order', type=int, help="Polynomial order for ChevConv", default=k_order)
    parser.add_argument('--hidden_channels', type=int, help="Hidden dimension", default=hidden_channels)
    parser.add_argument('--num_linear_layers', type=int, help="Number of linear layers", default=n_linear)
    parser.add_argument('--num_conv_layers', type=int, help="Number of convolution layers", default=n_conv)
    parser.add_argument('--pooling_type', type=str, help="Pooling method", default=pool)

    parser.add_argument('--batch_size', type=int, help="Batch size", default=batch_size)
    parser.add_argument('--lr', type=float, help="Learning rate", default=lr)
    parser.add_argument('--wd', type=float, help="Weight decay", default=wd)
    parser.add_argument('--num_epochs', type=int, help="Number of epochs to run", default=num_epochs)

    parser.add_argument('--num_classes', type=int, help="Number of classes", default=2)
    parser.add_argument('--num_node_features', type=int, help="Number of node features", default=7)
    parser.add_argument('--num_edge_attr', type=int, help="Number of edge attributes", default=4)

    return parser.parse_args()


def get_stats(array):
    return np.max(array), np.argmax(array), np.mean(array), np.var(array), np.min(array[::-1]), np.argmin(array[::-1])


if __name__ == "__main__":
    exp_models = ["ChebNet"]  # ["SAGE", "GCN", "GAT", "ChebNet", "EGC", "GIN"]
    exp_batch_sizes = [24, 32, 48, 64]
    exp_hidden_channels = [24, 48]  # [24, 48, 64]
    exp_lrs = [3e-3]  # , 1e-3]  # [5e-4, 1e-3, 5e-3, 1e-2]
    exp_wds = [5e-3]  # [3e-3, 5e-5]
    exp_epochs = [90]  # , 150]  # [125, 200]
    k_ords = [2]  # [2, 3, 4]
    num_lins = [1]  # [1, 2]
    num_convs = [3, 5, 7, 9, 11]  # [2, 3, 4, 5]
    poolings = ["add", "mean", "max"]
    results = []
    demo_run = None

    params_combinations = list(itertools.product(
        exp_models[:demo_run],
        exp_batch_sizes[:demo_run],
        exp_hidden_channels[:demo_run],
        exp_lrs[:demo_run],
        exp_wds[:demo_run],
        exp_epochs[:demo_run],
        k_ords[:demo_run],
        num_lins[:demo_run],
        num_convs[:demo_run],
        poolings[:demo_run]
    ))

    num_experiments = len(params_combinations)

    exp_idx = 0
    for experiment_params in params_combinations:
        args = parse_args(experiment_params)
        model_type, batch_size, hidden_channels, lr, wd, num_epochs, k_order, n_linear, n_conv, pool = experiment_params
        config.args = args

        train_acc, val_acc = [], []

        from experiments import run_experiment

        exp_idx += 1
        if exp_idx % 15 == 0 or exp_idx <= 5:
            print("\n=====================================================")
            print(f"\nExperiment [{exp_idx} / {num_experiments}] ->"
                  f"\n\tmodel = {args.model_type}\n\thidden_layers = {args.hidden_channels}"
                  f"\n\tnum_conv_layers = {args.num_conv_layers}"
                  f"\n\tnum_linear_layers = {args.num_linear_layers}"
                  f"\n\tk_order = {args.k_order}\n\tpooling_type = '{args.pooling_type}'\n"
                  f"\n\tnum_epochs = {args.num_epochs}\n\tbatch_size = {args.batch_size}"
                  f"\n\tlearning_rate = {args.lr}\n\tweight_decay = {args.wd}\n\n")

        train_loss, val_loss, train_acc, val_acc = run_experiment()
        (train_max_acc, train_max_acc_epoch, train_mean_acc, train_acc_var,
         train_min_acc, train_min_acc_epoch) = get_stats(train_acc)
        (val_max_acc, val_max_acc_epoch, val_mean_acc, val_acc_var,
         val_min_acc, val_min_acc_epoch) = get_stats(val_acc)

        results.append({
            'model': args.model_type,
            'num_conv_layers': args.num_conv_layers,
            'num_linear_layers': args.num_linear_layers,
            'polynomial_k_order': args.k_order,
            'pooling_method': args.pooling_type,
            'num_epochs': args.num_epochs,
            'batch_size': args.batch_size,
            'hidden_channels': args.hidden_channels,
            'lr': args.lr,
            'weight_decay': args.wd,
            'mean_train_accuracy': train_mean_acc,
            'var_train_accuracy': train_acc_var,
            'max_train_accuracy': train_max_acc,
            'max_train_accuracy_epoch': train_max_acc_epoch,
            'min_train_accuracy': train_min_acc,
            'min_train_accuracy_epoch': train_min_acc_epoch,
            'mean_val_accuracy': val_mean_acc,
            'var_val_accuracy': val_acc_var,
            'max_val_accuracy': val_max_acc,
            'max_val_accuracy_epoch': val_max_acc_epoch,
            'min_val_accuracy': val_min_acc,
            'min_val_accuracy_epoch': val_min_acc_epoch,
            'train_mean_loss': np.mean(train_loss),
            'val_mean_loss': np.mean(val_loss),
            'train_acc_array': train_acc,
            'val_acc_array': val_acc,
            'train_loss_array': train_loss,
            'val_loss_array': val_loss
        })
        if exp_idx % 15 == 0 or exp_idx <= 5:
            print(
                f"-> Train mean loss = {np.mean(train_loss):.4f} | Validation mean loss = {np.mean(val_loss):.4f}")
            print(
                f"-> Train max accuracy = {train_max_acc:.4f} | Validation max accuracy = {val_max_acc:.4f}")
            print("\n=====================================================")
        sleep(1)

    df_results = pd.DataFrame(results)
    df_results.to_csv('../results/experimentsFull_ChebNet_results.csv', index=False)
    print("Experiments results saved to results/experimentsFull_ChebNet_results.csv")
