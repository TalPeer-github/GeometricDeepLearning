import torch
from src.datasets_class.custom_graph_dataset import CustomGraphDataset
from sklearn import metrics
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
import os
import csv
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


def load_data(split: str, path="./data/", prints=False):
    file_path = f"{path}{split.strip()}.pt"
    dataset = torch.load(file_path)

    if prints:
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
            print(data.edge_attr)
            print(data.edge_index)
            print(data.x)
            for i, amax in enumerate(data.x.argmax(axis=1)):
                print(f"Node {i} : {amax} (index of the value 1, rest is 0)")

    return dataset


def save_graph_info(dataset, split='train'):
    graph_info = []
    node_info = []
    edge_info = []

    for i in range(len(dataset)):
        data = dataset[i]

        graph_info.append({
            'graph_index': i,
            'num_nodes': data.num_nodes,
            'num_edges': data.num_edges,
            'avg_node_degree': data.num_edges / data.num_nodes
        })

        nodes_type = data.x.argmax(axis=1)
        node_types_frequency = {f"Atom type {i}": nt.item() for i, nt in enumerate(nodes_type.bincount())}
        node_info.append({
            'graph_index': i,
            'node_types_frequency': node_types_frequency,
        })

        edges = data.edge_index.t().reshape(-1, 2)
        edges = [edge.numpy().tolist() for edge in edges]
        edges_type = data.edge_attr.argmax(axis=1)
        edges_type_dict = {f"{edge}": amax.item() for edge, amax in zip(edges, edges_type)}
        edge_types_frequency = {f"Bond type {i}": nt.item() for i, nt in enumerate(edges_type.bincount())}
        edge_info.append({
            'graph_index': i,
            'edge_types_frequency': edge_types_frequency,
            'edges_types': edges_type_dict
        })

    graph_df = pd.DataFrame(graph_info)
    node_df = pd.DataFrame(node_info)
    edge_df = pd.DataFrame(edge_info)

    df = pd.merge(graph_df, node_df, on='graph_index', how='outer')
    df = pd.merge(df, edge_df, on='graph_index', how='outer')
    if split != 'test':
        df['graph_label'] = df['graph_index'].apply(lambda i: dataset[int(i)].y.numpy().item())

    df.to_csv(f'../data/graphs_info/{split}_graph_info.csv', index=False)


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


def evaluate_metrics(y_true, y_pred, scores, output_file='../results/eval_results.csv', exp_args=None):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr, tpr, thresholds = roc_curve(y_true, scores, pos_label=1)
    auc_score = auc(fpr, tpr)
    exp_params_dict = vars(exp_args) if exp_args is not None else {}
    eval_metrics = {**exp_params_dict, "TN": tn, "FP": fp, "FN": fn, "TP": tp, "FPR": fpr, "TPR": tpr,
                    "thresholds": thresholds, "auc_score": auc_score}

    # exp_params_dict = vars(exp_args) if exp_args is not None else {}
    # exp_params_dict = {**exp_params_dict}
    # eval_metrics = exp_params_dict.update({"TN": tn, "FP": fp, "FN": fn, "TP": tp, "FPR": fpr, "TPR": tpr, "thresholds": thresholds,
    #      'auc_score': auc_score})

    file_exists = os.path.exists(output_file)
    f_mode = "a" if file_exists else "a"

    with open(output_file, mode='a', newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            header = list(eval_metrics.keys())
            writer.writerow(header)
        writer.writerow(eval_metrics.values())
