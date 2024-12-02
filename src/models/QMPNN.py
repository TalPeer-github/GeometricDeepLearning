from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool


class QMPNN(MessagePassing):
    def __init__(self, node_input_dim, edge_input_dim, hidden_dim, num_classes):
        super(QMPNN, self).__init__(aggr='add')  # Use "add" aggregation for message passing
        self.node_embedding = nn.Embedding(node_input_dim, hidden_dim)
        self.edge_embedding = nn.Embedding(edge_input_dim, hidden_dim)

        # Message passing layers (Graph Convolution layers)
        self.message_fc = nn.Linear(2 * hidden_dim, hidden_dim)  # Combine node and edge features
        self.update_fc = nn.Linear(hidden_dim, hidden_dim)  # Update node feature after passing messages

        # Readout layer to aggregate node features to graph-level features
        self.pooling = global_mean_pool  # Global mean pooling (you can change this)

        # Fully connected layers for classification
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        # Extract features
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # Ensure the node indices are integers (required for embedding)
        x = x.to(torch.long)  # Cast node feature tensor to LongTensor

        # Embed nodes and edges
        x = self.node_embedding(x)  # Node feature embedding
        edge_attr = self.edge_embedding(edge_attr)  # Edge feature embedding

        # Handle edge indices for batched graphs
        # In batched graphs, edge_index should be used with the 'batch' information to map across graphs
        batch = data.batch  # This tells which graph each node belongs to
        num_graphs = int(batch.max()) + 1  # The number of graphs in the batch

        # We don't need to modify edge_index manually since PyTorch Geometric's `message_passing` will take care of that,
        # but we should ensure that node features and edge indices are properly aligned across the batch.

        # Perform message passing
        x = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        # Pool node features to get graph-level representation
        x = self.pooling(x, batch)

        # Fully connected layers for classification
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)

    def message(self, x_j, edge_attr):
        """

        :param x_j:
        :param edge_attr:
        :return:
        """
        # Combine node and edge features (message passing step)
        return F.relu(self.message_fc(torch.cat([x_j, edge_attr], dim=-1)))

    def update(self, aggr_out):
        # Update node features (aggregation step)
        return F.relu(self.update_fc(aggr_out))
