import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from src.models.GCN import *
from src.models.SAGE import *
from src.models.GAT import *
from src.models.QMPNN import *
from src.models.EGC import *
from src.models.cheb_net import *
from utils.data_utils import load_data, loader_details, evaluate_metrics
import config
from utils.viz import *

import os
import csv

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def explore_data(graphs_data, split='train', n=30):
    for i in range(len(graphs_data)):
        if i % n == 0:
            visualize_graph(graphs_data[i])
            # visualize_graph_3d(graphs_data[i])
            # visualize_graph_shell(graphs_data[i])


def _train(model, train_loader, optimizer, criterion):
    """
     Iterate in batches over the training dataset.
     Perform a single forward pass
     Compute the loss.
     Derive gradients
     Update parameters based on gradients
     Clear gradients

    """
    model.train()
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()


@torch.no_grad()
def test(model, loader, criterion):
    """
    Iterate in batches over the training/test dataset.
    Use the class with the highest probability.
    Check against ground-truth labels.
    Derive ratio of correct predictions.
    :param loader:
    :return:
    """
    model.eval()
    total_loss = 0.0
    correct = 0

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            logits = model(data.x, data.edge_index, data.batch)
            loss = criterion(logits, data.y)
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += int((pred == data.y).sum())

    avg_loss = total_loss / len(loader)
    accuracy = correct / len(loader.dataset)
    return avg_loss, accuracy


@torch.no_grad()
def predict_test(model, loader, output_file="predictions.txt", include_score=True):
    args = config.args

    output_file = output_file if args.env != 'local' else f'../results/{output_file}'

    model.eval()
    preds = []
    confidences = []
    y_true = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            logits = model(data.x, data.edge_index, data.batch)
            prob = F.softmax(logits, dim=1)
            pred = prob.argmax(dim=1)
            conf = prob.max(dim=1).values

            preds.append(pred.cpu().numpy())
            confidences.append(conf.cpu().numpy())
            y_true.extend(data.y)

    if args.run_mode == 'eval':
        evaluate_metrics(y_true=y_true, y_pred=preds, scores=confidences, exp_args=args)
    if args.env != 'local':

        with open(output_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            if include_score:
                writer.writerow(["label", "score"])
                writer.writerows(zip(preds, confidences))
            else:
                writer.writerow(["label"])
                writer.writerows([[p] for p in preds])


def run_experiment():
    args = config.args

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    explore_graphs = False
    follow_training = False
    create_pseudo_test = True and (args.env == 'local')

    data_dir = "../data/" if args.env == 'local' else "./data/"
    train_dataset = load_data('train', path=data_dir)
    val_dataset = load_data('val', path=data_dir)
    if create_pseudo_test:
        train_dataset, test_dataset = train_dataset[:125], train_dataset[125:]
    else:
        test_dataset = load_data('test', path=data_dir)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size // 2, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    if explore_graphs:
        explore_data(train_dataset)
        # loader_details(train_loader, split='train')
        # loader_details(val_loader, split='val')

    if args.model_type == "GCN" or args.model_type is None:
        model = GCN(num_node_features=args.num_node_features, hidden_channels=args.hidden_channels,
                    num_classes=args.num_classes).to(device)
    elif args.model_type == "SAGE":
        model = GraphSAGE(num_node_features=args.num_node_features, hidden_channels=args.hidden_channels,
                          num_classes=args.num_classes)
    elif args.model_type == "GAT":
        model = GAT(num_node_features=args.num_node_features, hidden_channels=args.hidden_channels,
                    num_classes=args.num_classes)
    elif args.model_type == "EGC":
        model = EGCNet(num_node_features=args.num_node_features, hidden_channels=args.hidden_channels,
                       num_classes=args.num_classes)
    elif args.model_type == "ChebNet":
        model = ChebNet(num_node_features=args.num_node_features, hidden_channels=args.hidden_channels,
                        num_classes=args.num_classes, num_lin=args.num_linear_layers, num_conv=args.num_conv_layers,
                        k_order=args.k_order, pooling_type=args.pooling_type)
    else:
        model = GCN(num_node_features=args.num_node_features, hidden_channels=args.hidden_channels,
                    num_classes=args.num_classes)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    criterion = torch.nn.CrossEntropyLoss()

    train_accuracies, val_accuracies = [], []
    train_losses, val_losses = [], []

    for epoch in range(1, args.num_epochs + 1):
        _train(model, train_loader, optimizer, criterion)

        train_loss, train_acc = test(model, train_loader, criterion)
        val_loss, val_acc = test(model, val_loader, criterion)

        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if create_pseudo_test and (val_acc < 0.7 and epoch >= 40) or follow_training:
            if not follow_training:
                model.print_params()
            follow_training = True
            print(f"Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    if create_pseudo_test and (follow_training
                               or (np.max(val_accuracies) >= 0.9
                                   or np.min(val_accuracies) < 0.7
                                   or np.mean(np.abs(np.array(train_accuracies) - np.array(val_accuracies))) >= 0.2)):
        plot_learning_curve(train_accuracies, val_accuracies, train_losses, val_losses)

    if args.run_mode == 'eval' and create_pseudo_test:
        test_loss, test_acc = test(model, test_loader, criterion)
        predict_test(model=model, loader=test_loader)
        return train_losses, val_losses, train_accuracies, val_accuracies, test_loss, test_acc
    elif args.run_mode == 'eval' and not create_pseudo_test:
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
        predict_test(model=model, loader=val_loader)
    elif args.run_mode == 'exp' and create_pseudo_test:
        test_loss, test_acc = test(model, test_loader, criterion)
        plot_learning_curve(train_accuracies, val_accuracies, train_losses, val_losses)
        return train_losses, val_losses, train_accuracies, val_accuracies, test_loss, test_acc
    elif args.env != 'local':
        predict_test(model=model, loader=test_loader)
    return train_losses, val_losses, train_accuracies, val_accuracies, 0, 0
