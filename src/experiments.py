import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from src.models.GCN import *
from src.models.SAGE import *
from src.models.GAT import *
from src.models.QMPNN import *
from src.models.EGC import *
from src.models.CHEV import *
from utils.data_utils import load_data, loader_details
import config
from utils.viz import *
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
        data = data.to(device)
        logits = model(data.x, data.edge_index, data.batch)
        pred = logits.argmax(dim=1)
        correct += int((pred == data.y).sum())
    return correct / len(loader.dataset)


def qmpnn_train(train_loader):
    model.train()
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output, batch.y)
        loss.backward()
        optimizer.step()


def run_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = config.args
    explore_graphs = False

    splits = ['train', 'val', 'test']
    train_dataset = load_data('train')
    val_dataset = load_data('val')
    test_dataset = load_data('test')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

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
    elif args.model_type == "QMPNN":
        model = QMPNN(node_input_dim=args.num_node_features, edge_input_dim=args.num_edge_attr,
                      hidden_dim=args.hidden_channels, num_classes=args.num_classes)
    elif args.model_type == "EGC":
        model = EGCNet(num_node_features=args.num_node_features, hidden_channels=args.hidden_channels,
                       num_classes=args.num_classes)
    elif args.model_type == "CHEV":
        model = ChebNet(num_node_features=args.num_node_features, hidden_channels=args.hidden_channels,
                        num_classes=args.num_classes)
    else:
        model = GCN(num_node_features=args.num_node_features, hidden_channels=args.hidden_channels,
                    num_classes=args.num_classes)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    criterion = torch.nn.CrossEntropyLoss()
    train_accuracies, val_accuracies = [], []

    for epoch in range(1, args.num_epochs + 1):
        if not args.model_type == "QMPNN":
            _train(model, train_loader, optimizer, criterion)
        else:
            qmpnn_train(train_loader)
        train_acc = round(test(model, train_loader), 6)
        val_acc = round(test(model, val_loader), 6)

        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        if epoch % 20 == 0:
            print(f"Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    # plot_learning_curve(train_accuracies, val_accuracies)
    return train_accuracies, val_accuracies
