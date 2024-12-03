import matplotlib.pyplot as plt
import networkx as nx
import config
import torch
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.decomposition import PCA

args = config.args


def assign_nodes_colors(node_features):
    node_features = torch.argmax(node_features, dim=1).numpy()

    color_map = {
        0: '#B7E0FF',  # Hydrogen #light blue
        1: '#E78F81',  # Carbon # light red
        2: '#B1AFFF',  # Nitrogen #purpulish
        3: '#5D9C59',  # Oxygen # green
        4: '#F8BDEB',  # Fluorine # pinkish
        5: '#F0B86E',  # Phosphorus # orange
        6: '#FEFF86'  # Sulfur # Yellow
    }

    node_colors = [color_map[atom_type] for atom_type in node_features]

    return node_colors


def generate_3d_positions(features):
    pca = PCA(n_components=3)
    positions = pca.fit_transform(features)
    return {i: pos for i, pos in enumerate(positions)}


def visualize_graph_shell(graph):
    edge_index = graph.edge_index
    edges = edge_index.t().tolist()

    G = nx.Graph()
    G.add_edges_from(edges)

    node_features = graph.x
    for i, features in enumerate(node_features):
        G.nodes[i]['features'] = features.tolist()

    # Calculate degrees of each node
    degrees = {node: G.degree(node) for node in G.nodes}

    # Sort nodes based on degree (highest degree first)
    sorted_nodes_by_degree = sorted(degrees.items(), key=lambda x: x[1], reverse=True)

    # Divide the nodes into shells based on their degree
    shells = [[] for _ in range(len(sorted_nodes_by_degree))]
    for i, (node, _) in enumerate(sorted_nodes_by_degree):
        shells[i % len(shells)].append(node)  # Distribute nodes in shells in a round-robin manner

    # Use the shell layout
    pos = nx.shell_layout(G, nlist=shells)

    # Adjusting the center: translate positions so that the graph's center is in the middle
    pos = {node: (x - np.mean(list(x for x, y in pos.values())),
                  y - np.mean(list(y for x, y in pos.values())))
           for node, (x, y) in pos.items()}

    plt.figure(figsize=(8, 8))

    if graph.y.numpy().item() == 0:
        node_color, edge_color = '#DA7297', '#FFB4C2'
    else:
        node_color, edge_color = '#DE8F5F', '#FFCF9D'

    # Assign colors based on node features (you may have a custom color function)
    nodes_colors_ = assign_nodes_colors(node_features)

    # Draw graph using nx.draw with the positions from shell_layout
    nx.draw(G, pos, with_labels=True, node_size=500, node_color=nodes_colors_, font_size=10, font_weight='bold',
            edge_color=edge_color)

    # Label the nodes with their features
    node_labels = {i: f"\n\n{str(node_features[i].tolist())}" for i in range(len(node_features))}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)

    # Title and layout
    plt.suptitle(f'Label = {graph.y.numpy().item()}', fontweight='bold')
    plt.tight_layout()
    plt.show()


def visualize_graph(graph):
    edge_index = graph.edge_index
    edges = edge_index.t().tolist()

    G = nx.Graph()
    G.add_edges_from(edges)

    node_features = graph.x
    for i, features in enumerate(node_features):
        G.nodes[i]['features'] = features.tolist()

    pos = nx.spring_layout(G, seed=42)

    plt.figure(figsize=(8, 8))
    if graph.y.numpy().item() == 0:
        node_color, edge_color = '#DA7297', '#FFB4C2'
    else:
        node_color, edge_color = '#DE8F5F', '#FFCF9D'

    nodes_colors_ = assign_nodes_colors(node_features)
    nx.draw(G, pos, with_labels=True, node_size=500, node_color=nodes_colors_, font_size=10, font_weight='bold',
            edge_color=edge_color)

    node_labels = {i: f"\n\n{str(node_features[i].tolist())}" for i in range(len(node_features))}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)

    plt.suptitle(f'Label = {graph.y.numpy().item()}', fontweight='bold')
    plt.tight_layout()
    plt.show()


def visualize_graph_3d(graph):
    """
    Visualize a graph in 3D with colored nodes and edges.

    Parameters:
        graph: PyG Data object containing edge_index, x (node features), and y (label).
    """
    edge_index = graph.edge_index
    edges = edge_index.t().tolist()

    G = nx.Graph()
    G.add_edges_from(edges)

    node_features = graph.x
    for i, features in enumerate(node_features):
        G.nodes[i]['features'] = features.tolist()

    # random layout 3D positions for nodes
    pos = {i: np.random.rand(3) for i in range(len(node_features))}
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    if graph.y.numpy().item() == 0:
        node_color, edge_color = '#DA7297', '#FFB4C2'
    else:
        node_color, edge_color = '#DE8F5F', '#FFCF9D'

    nodes_colors_ = assign_nodes_colors(node_features)

    for edge in edges:
        start, end = edge
        start_pos = pos[start]
        end_pos = pos[end]
        ax.plot(
            [start_pos[0], end_pos[0]],
            [start_pos[1], end_pos[1]],
            [start_pos[2], end_pos[2]],
            color='#A6AEBF',
            alpha=0.7,
            linewidth=2,
        )

    for idx, coords in pos.items():
        ax.scatter(
            coords[0], coords[1], coords[2],
            color=nodes_colors_[idx],
            s=400,
            edgecolors='k',
            label=str(idx)
        )

    for idx, coords in pos.items():
        ax.text(
            coords[0], coords[1], coords[2],
            s=str(idx),
            fontsize=8,
            ha='center'
        )

    num_nodes = graph.num_nodes
    num_edges = graph.num_edges
    mean_node_deg = num_nodes / num_edges
    param_text = f"Number of nodes: {num_nodes}\nNumber of edges: {num_edges}\n"
    fig.text(
        0.8, 1,  # x, y position in figure coordinates (normalized, 0 to 1)
        param_text,
        fontsize=10,
        verticalalignment='center',
        horizontalalignment='left',
        bbox=dict(facecolor='#FEFAF6', alpha=0.95, edgecolor='#102C57', boxstyle='round,pad=0.5')
    )

    ax.set_title(f'3D Visualization of Graph (Label = {graph.y.numpy().item()})', fontweight='bold')
    ax.grid(False)
    plt.tight_layout()
    plt.show()


def visualize_graph_3d_pca(graph):
    """
    Visualize a graph in 3D with colored nodes and edges.

    Parameters:
        graph: PyG Data object containing edge_index, x (node features), and y (label).
    """
    edge_index = graph.edge_index
    edges = edge_index.t().tolist()

    G = nx.Graph()
    G.add_edges_from(edges)

    node_features = graph.x
    for i, features in enumerate(node_features):
        G.nodes[i]['features'] = features.tolist()

    pos = generate_3d_positions(node_features.numpy())
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    if graph.y.numpy().item() == 0:
        node_color, edge_color = '#DA7297', '#FFB4C2'
    else:
        node_color, edge_color = '#DE8F5F', '#FFCF9D'

    nodes_colors_ = assign_nodes_colors(node_features)

    edge_colors = ['#FF5733', '#33FF57', '#3357FF', '#FFC300']  # Colors for different bond types
    for edge, attr in zip(edges, graph.edge_attr.numpy()):
        start, end = edge
        bond_type = np.argmax(attr)  # Assuming one-hot encoding
        ax.plot(
            [pos[start][0], pos[end][0]],
            [pos[start][1], pos[end][1]],
            [pos[start][2], pos[end][2]],
            color=edge_colors[bond_type],
            alpha=0.7,
            linewidth=2,
        )

    for idx, coords in pos.items():
        ax.scatter(
            coords[0], coords[1], coords[2],
            color=nodes_colors_[idx],
            s=400,
            edgecolors='k',
            label=str(idx)
        )

    for idx, coords in pos.items():
        ax.text(
            coords[0], coords[1], coords[2],
            s=str(idx),
            fontsize=8,
            ha='center'
        )

    num_nodes = graph.num_nodes
    num_edges = graph.num_edges
    mean_node_deg = num_nodes / num_edges
    param_text = f"Number of nodes: {num_nodes}\nNumber of edges: {num_edges}\n"
    fig.text(
        0.8, 0.5,  # x, y position in figure coordinates (normalized, 0 to 1)
        param_text,
        fontsize=10,
        verticalalignment='center',
        horizontalalignment='left',
        bbox=dict(facecolor='#FEFAF6', alpha=0.95, edgecolor='#102C57', boxstyle='round,pad=0.5')
    )

    ax.set_title(f'3D Visualization of Graph (Label = {graph.y.numpy().item()})', fontweight='bold')
    plt.show()


def get_file_path(split='train'):
    file_path = f'../data/{split}.pt'
    return file_path


def plot_learning_curve(train_accuracies, val_accuracies, train_losses, val_losses, model_type=args.model_type,
                        num_epochs=args.num_epochs,
                        batch_size=args.batch_size, lr=args.lr, hidden_dim=args.hidden_channels,
                        plot_losses=True, save_fig=True):
    args = config.args
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(16, 5))

    if plot_losses:
        n_rows, n_cols, index_pos = 1, 2, 1
    else:
        n_rows, n_cols, index_pos = 1, 1, 1
    val_color = "#C55300" #"#820000"  # "#8B322C"
    train_color = "#0A516D" #"#102C57"  # "#102C57"
    plt.subplot(n_rows, n_cols, index_pos)
    plt.plot(epochs, train_accuracies, label='Train Accuracy', color=train_color, linewidth=2, alpha=1.)
    plt.scatter(epochs[::5], train_accuracies[::5], color=train_color, linewidth=0.1, alpha=0.5)
    plt.plot(epochs, val_accuracies, label='Validation Accuracy', color=val_color, linewidth=2, alpha=1.)
    plt.scatter(epochs[::5], val_accuracies[::5], color=val_color, linewidth=0.1, alpha=0.5)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'\n\n\n')  # Accuracy Over Epochs', fontweight='bold')
    plt.legend()

    val_mean_acc = sum(val_accuracies) / len(val_accuracies)
    train_mean_acc = sum(train_accuracies) / len(train_accuracies)

    if plot_losses:
        plt.subplot(n_rows, n_cols, 2)
        plt.plot(epochs, train_losses, label='Train Loss', color=train_color, linewidth=2, alpha=1.)
        plt.plot(epochs, val_losses, label='Validation Loss', color=val_color, linewidth=2, alpha=1.)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'\n\n\n')

        plt.legend()

    if model_type == "ChebNet":
        param_text = (f"{args.num_conv_layers} Conv layers | {args.num_linear_layers} Linear layers "
                      f"| '{args.pooling_type}' Pooling | Order of Chebyshev Polynomials = {args.k_order}")
    else:
        param_text = (
            f"Model: {model_type}\n\nNum Epochs = {num_epochs}\nBatch Size = {batch_size}\nLR = {lr}\nHidden Dim = {hidden_dim}"
            f"\n\nTrain Mean Accuracy = {train_mean_acc:.4f}\nValidation Mean Accuracy = {val_mean_acc:.4f}")

    plt.gcf().text(
        0.5, 0.975,
        args.model_type,
        fontsize=15,
        verticalalignment='top',
        horizontalalignment='center',
        fontweight='bold',
        bbox=dict(facecolor='#FEFAF6', alpha=0.95, edgecolor=train_color, boxstyle='round,pad=0.3')
    )
    plt.gcf().text(
        0.5, 0.90,
        param_text,
        fontsize=10,
        verticalalignment='top',
        horizontalalignment='center',
        bbox=dict(facecolor='#FEFAF6', alpha=0.95, edgecolor=train_color, boxstyle='round,pad=0.5')
    )
    plt.legend()

    plt.tight_layout()

    if save_fig:
        file_name = f"val_acc_{val_mean_acc:.4f}".replace('.', '')
        if model_type == "ChebNet":
            img_path = f'../results/images/{model_type}_{file_name}_k{args.k_order}_nlin{args.num_linear_layers}_nconv{args.num_conv_layers}_{args.pooling_type}Pooling.png'
        else:
            img_path = f'../results/images/{model_type}_{file_name}_lr{args.lr}_batch{args.batch_size}_hc{args.hidden_channels}.png'
        plt.savefig(img_path)

    plt.show()


ask_split = False
explore = False
only_viz = False
if only_viz:
    if ask_split:
        split = input("Required Split [train/eval/test] -> ").strip()
    else:
        split = "train"
    split_path = get_file_path(split)
    graphs_data = load_data(path=split_path, split=split)
    if explore:
        explore_data(graphs_data, split=split)
