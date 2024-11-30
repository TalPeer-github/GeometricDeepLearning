import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from src.models.GCN import *
from src.models.SAGE import *
from src.models.GAT import *
from utils.data_utils import load_data, loader_details
import config
from utils.viz import plot_learning_curve, visualize_graph
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
args = config.args


def explore_data(graphs_data, split='train', n=10):
    for i in range(len(graphs_data)):
        if i % n == 0:
            visualize_graph(graphs_data[i])


def train(model, train_loader, optimizer, criterion):
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
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def test(model, loader):
    """
    Iterate in batches over the training/test dataset.
    Use the class with the highest probability.
    Check against ground-truth labels.
    Derive ratio of correct predictions.
    :param loader:
    :return:
    """
    model.eval()
    correct = 0
    for data in loader:
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
    return correct / len(loader.dataset)


def start_experiments():
    explore_graphs = False

    splits = ['train', 'val', 'test']
    train_dataset = load_data('train')
    val_dataset = load_data('val')
    test_dataset = load_data('test')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    loader_details(train_loader, split='train')
    loader_details(val_loader, split='val')
    if explore_graphs:
        explore_data(train_dataset)
        explore_data(val_dataset)

    if args.model_type == "GCN" or args.model_type is None:
        model = GCN(num_node_features=args.num_node_features, hidden_channels=args.hidden_channels,
                    num_classes=args.num_classes)
    elif args.model_type == "SAGE":
        model = GraphSAGE(num_node_features=args.num_node_features, hidden_channels=args.hidden_channels,
                          num_classes=args.num_classes)
    elif args.model_type == "GAT":
        model = GAT(num_node_features=args.num_node_features, hidden_channels=args.hidden_channels,
                    num_classes=args.num_classes)
    else:
        model = GCN(num_node_features=args.num_node_features, hidden_channels=args.hidden_channels,
                    num_classes=args.num_classes)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    train_accuracies, val_accuracies = [], []

    for epoch in range(1, args.num_epochs + 1):
        train(model, train_loader, optimizer, criterion)
        train_acc = test(model, train_loader)
        val_acc = test(model, val_loader)

        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(f"Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    plot_learning_curve(train_accuracies, val_accuracies)
    # if np.mean(val_accuracies) > best_acc:
    #     model_path = 'path_to_save_model.pth'
    #     torch.save(model.state_dict(), model_path)
