import matplotlib.pyplot as plt
import networkx as nx
import config
import torch


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


def get_file_path(split='train'):
    file_path = f'../data/{split}.pt'
    return file_path


def plot_learning_curve(train_accuracies, val_accuracies, model_type=args.model_type, num_epochs=args.num_epochs,
                        batch_size=args.batch_size, lr=args.lr, hidden_dim=args.hidden_channels,
                        plot_losses=False, save_fig=True):
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(10, 5))

    if plot_losses:
        n_rows, n_cols, index_pos = 1, 2, 1
    else:
        n_rows, n_cols, index_pos = 1, 1, 1
    plt.subplot(n_rows, n_cols, index_pos)
    plt.plot(epochs, train_accuracies, label='Train Accuracy', color="#102C57", linewidth=2)
    plt.plot(epochs, val_accuracies, label='Validation Accuracy', color="#8B322C", linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.legend()

    val_mean_acc = sum(val_accuracies) / len(val_accuracies)
    train_mean_acc = sum(train_accuracies) / len(train_accuracies)
    param_text = (
        f"Model: {model_type}\n\nNum Epochs = {num_epochs}\nBatch Size = {batch_size}\nLR = {lr}\nHidden Dim = {hidden_dim}"
        f"\n\nTrain Mean Accuracy = {train_mean_acc:.4f}\nValidation Mean Accuracy = {val_mean_acc:.4f}")
    plt.gca().text(0.15, 0.15, param_text, transform=plt.gca().transAxes, fontsize=8, verticalalignment='center',
                   horizontalalignment='center', bbox=dict(facecolor='#FEFAF6', alpha=0.95, edgecolor='#102C57',
                                                           boxstyle='round,pad=0.5'))

    if plot_losses:
        plt.subplot(1, 2, 2)
        plt.plot(epochs, [0.0] * num_epochs, label='Train Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Epochs')
        plt.legend()

    plt.tight_layout()
    if save_fig:
        file_name = f"val_acc_{val_mean_acc:.4f}".replace('.', '')
        img_path = f'../results/images/{model_type}_{file_name}_lr{args.lr}_batch{args.batch_size}_hc{args.hidden_channels}_.png'
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
