from torch_geometric.loader import DataLoader
from config.local_config import *
from src.models.GCN import *
from src.models.SAGE import *
from src.models.GAT import *
from src.utils.viz import plot_learning_curve

import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def load_data(split: str, path="../data/"):
    file_path = f"{path}{split.strip()}.pt"
    dataset = torch.load(file_path)

    if isinstance(dataset, dict):
        print("Keys:", dataset.keys())
    else:
        print(f"Split = {split.capitalize()} Dataset")
        print("Loaded data:", dataset)
        print(f"Data Object Class : {type(dataset)}\n")
        print(f'Number of graphs: {len(dataset)}')
        print(f'Number of features: {dataset.num_features}')
        print(f'Number of classes: {num_classes} -> [0,1]')

        data = dataset[0]

        print()
        print(data)
        print('=============================================================')

        print(f'Number of nodes: {data.num_nodes}')
        print(f'Number of edges: {data.num_edges}')
        print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
        print(f'Has isolated nodes: {data.has_isolated_nodes()}')
        print(f'Has self-loops: {data.has_self_loops()}')
        print(f'Is undirected: {data.is_undirected()}')

    return dataset


def loader_details(data_loader, split: str):
    """
    Each `Batch` object is equipped with a `batch vector`, which maps each node to its respective graph in the batch.
    :param data_loader: Data iterable
    :param split: dataset split - train/val/test
    """
    print(f"Split = {split.capitalize()} Dataset\n")
    for batch_idx, data in enumerate(data_loader):
        print(f'Batch {batch_idx + 1}:')
        print('=======')
        print(f'Number of graphs in the current batch: {data.num_graphs}')
        print(data)
        print()


def train():
    """
     Iterate in batches over the training dataset.
     Perform a single forward pass
     Compute the loss.
     Derive gradients
     Update parameters based on gradients
     Clear gradients
    :return:
    """

    model.train()
    for data in train_loader:
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def test(data_loader: torch.utils.data.DataLoader):
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
    for data in data_loader:
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)
        actual = torch.tensor(data.y.numpy())
        correct += int(torch.sum(torch.eq(pred, actual)))
    return correct / len(data_loader.dataset)


args = config.args
splits = ['train', 'val', 'test']
train_dataset = load_data('train')
val_dataset = load_data('val')
test_dataset = load_data('test')

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

loader_details(train_loader, split='train')
loader_details(test_loader, split='test')

if model_type == "GCN" or model_type is None:
    model = GCN(num_node_features=args.num_node_features, hidden_channels=args.hidden_channels,
                num_classes=args.num_classes)
elif model_type == "SAGE":
    model = GraphSAGE(num_node_features=args.num_node_features, hidden_channels=args.hidden_channels,
                      num_classes=args.num_classes)
elif model_type == "GAT":
    model = GAT(num_node_features=args.num_node_features, hidden_channels=args.hidden_channels,
                num_classes=args.num_classes)
else:
    model = GCN(num_node_features=args.num_node_features, hidden_channels=args.hidden_channels,
                num_classes=args.num_classes)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
criterion = torch.nn.CrossEntropyLoss()

train_accuracies, val_accuracies = [], []
for epoch in range(1, num_epochs + 1):
    train()
    train_acc = test(train_loader)
    val_acc = test(val_loader)

    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)

    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f} | Validation Acc: {val_acc:.4f}')

plot_learning_curve(train_accuracies=train_accuracies, val_accuracies=val_accuracies, num_epochs=num_epochs)
