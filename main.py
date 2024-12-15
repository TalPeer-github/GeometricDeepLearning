import numpy as np
import itertools
import src.config
from argparse import ArgumentParser
import pandas as pd
from src.datasets_class.custom_graph_dataset import *
from time import sleep


def parse_args(exp_params):
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
    parser.add_argument('--seed', type=int, help="seed", default=42)
    return parser.parse_args()


def get_stats(array):
    return np.max(array), np.argmax(array), np.mean(array), np.var(array), np.min(array[::-1]), np.argmin(array[::-1])


if __name__ == "__main__":
    exp_models = ["ChebNet"]
    exp_batch_sizes = [24]
    exp_hidden_channels = [16]
    exp_lrs = [3e-3]
    exp_wds = [5e-3]
    exp_epochs = [90]
    k_ords = [3]
    num_lins = [1]
    num_convs = [5]
    poolings = ["max"]
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

    for experiment_params in params_combinations:
        args = parse_args(experiment_params)
        model_type, batch_size, hidden_channels, lr, wd, num_epochs, k_order, n_linear, n_conv, pool = experiment_params
        src.config.args = args

        train_acc, val_acc = [], []

        from src.experiments import run_experiment

        print("\n=====================================================")

        print(f"\nModel = {args.model_type} ->"
              f"\n\tNumber of Convolution Layers = {args.num_conv_layers}"
              f"\n\tNumber of Linear Layers = {args.num_linear_layers}"
              f"\n\tHidden Channels = {args.hidden_channels}"
              f"\n\tOrder of Chebyshev Polynomials = {args.k_order}\n\tPooling Type = '{args.pooling_type}'\n"
              f"\n\tTotal Epochs = {args.num_epochs}\n\tBatch Size = {args.batch_size}"
              f"\n\tOptimizer = torch.optim.Adam\n\tCriterion = torch.nn.CrossEntropyLoss"
              f"\n\tLearning Rate = {args.lr}\n\tWeight Decay = {args.wd}\n\n")

        train_loss, val_loss, train_acc, val_acc, _, _ = run_experiment()
        (train_max_acc, train_max_acc_epoch, train_mean_acc, train_acc_var,
         train_min_acc, train_min_acc_epoch) = get_stats(train_acc)
        (val_max_acc, val_max_acc_epoch, val_mean_acc, val_acc_var,
         val_min_acc, val_min_acc_epoch) = get_stats(val_acc)

        print(
            f"-> Train mean loss = {np.mean(train_loss):.4f} | Validation mean loss = {np.mean(val_loss):.4f}")
        print(
            f"-> Train mean accuracy = {train_mean_acc:.4f} | Validation mean accuracy = {val_mean_acc:.4f}")
        print(
            f"-> Train max accuracy = {train_max_acc:.4f} | Validation max accuracy = {val_max_acc:.4f}")

        output_file = "predictions.txt"
        print(f"\n\nPredictions saved to {output_file}")

        print("\n=====================================================")
