import numpy as np

import config
from argparse import ArgumentParser
import pandas as pd
from datasets_class.custom_graph_dataset import *
from time import sleep


def parse_args():
    """
    # ChevNet works will with LR 0.001 and num_epochs 120 weight_decay 5e-5 batch_size = 64 hidden_channels=24
    # EGC works will with LR 0.001 and num_epochs 120 weight_decay 5e-5 batch_size = 64 hidden_channels=24
    # SAGE works well with LR 0.01 and 125 Epochs batch_size = 32 hidden_channels=32
    # GAT works well with LR 0.0005 and 100 Epochs batch_size = 32 hidden_channels=32 # small LR in general
    :return:
    """
    model_type = "CHEV"
    batch_size = 32
    hidden_channels = 48
    lr = 5e-5
    wd = 3e-3
    num_epochs = 125
    k_order = 2

    parser = ArgumentParser(description="Graph Classification Task")
    parser.add_argument('--model_type', type=str, help="Chosen Model", default=model_type)
    parser.add_argument('--batch_size', type=int, help="Batch size", default=batch_size)
    parser.add_argument('--hidden_channels', type=int, help="Hidden dimension", default=hidden_channels)
    parser.add_argument('--lr', type=float, help="Learning rate", default=lr)
    parser.add_argument('--wd', type=float, help="Weight decay", default=wd)
    parser.add_argument('--num_epochs', type=int, help="Number of epochs to run", default=num_epochs)
    parser.add_argument('--k_order', type=int, help="Polynomial order for ChevConv", default=k_order)
    parser.add_argument('--num_classes', type=int, help="Number of classes", default=2)
    parser.add_argument('--num_node_features', type=int, help="Number of node features", default=7)
    parser.add_argument('--num_edge_attr', type=int, help="Number of edge attributes", default=4)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config.args = args

    from experiments import run_experiment

    _, _ = run_experiment()
