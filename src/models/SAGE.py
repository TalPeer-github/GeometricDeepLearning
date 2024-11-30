import torch
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import global_mean_pool


class GraphSAGE(torch.nn.Module):
    """
    GraphSAGE model using different aggregation methods to compute graph representations.
    """

    def __init__(self, num_node_features, hidden_channels, num_classes, p=0.4, seed=42):
        super(GraphSAGE, self).__init__()
        torch.manual_seed(seed=seed)
        self.p = p

        self.conv1 = SAGEConv(num_node_features, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        """
        Obtain node embeddings, apply aggregation, add readout layer, and apply the final classifier
        :param x: node features
        :param edge_index: graph connectivity
        :param batch: batch indices for graph-level pooling
        :return: graph-level predictions
        """

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)

        x = global_mean_pool(x, batch)

        x = F.dropout(x, p=self.p, training=self.training)
        x = self.lin(x)

        return x
