import torch
from src.datasets_class.custom_graph_dataset import CustomGraphDataset


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
