import config
from argparse import ArgumentParser
from datasets_class.custom_graph_dataset import *


def parse_args():
    """
    # SAGE works well with LR 0.01 and 125 Epochs batch_size = 32 hidden_channels=32
    # GAT works well with LR 0.0005 and 100 Epochs batch_size = 32 hidden_channels=32 # small LR in general
    :return:
    """
    batch_size = 32
    hidden_channels = 32
    lr = 0.002
    num_epochs = 125

    parser = ArgumentParser(description="Graph Classification Task")
    parser.add_argument('--model_type', type=str, help="Chosen Model", default="SAGE")
    parser.add_argument('--batch_size', type=int, help="Batch size", default=batch_size)
    parser.add_argument('--hidden_channels', type=int, help="Hidden dimension", default=hidden_channels)
    parser.add_argument('--lr', type=float, help="Learning rate", default=lr)
    parser.add_argument('--num_epochs', type=int, help="Number of epochs to run", default=num_epochs)

    parser.add_argument('--num_classes', type=int, help="Number of classes", default=2)
    parser.add_argument('--num_node_features', type=int, help="Number of node features", default=7)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config.args = args

    from experiments import start_experiments
    start_experiments()
