import torch
from torch_geometric.nn import GATConv
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import global_mean_pool

class GAT(torch.nn.Module):
    """
    GAT model using attention mechanisms to selectively aggregate neighbor features.
    """

    def __init__(self, num_node_features, hidden_channels, num_classes, num_heads=4, p=0.4, seed=42):
        super(GAT, self).__init__()
        torch.manual_seed(seed=seed)
        self.p = p
        self.num_heads = num_heads

        self.conv1 = GATConv(num_node_features, hidden_channels, heads=self.num_heads, concat=True)
        self.conv2 = GATConv(hidden_channels * self.num_heads, hidden_channels, heads=self.num_heads, concat=True)
        self.lin = Linear(hidden_channels * self.num_heads, num_classes)

    def forward(self, x, edge_index, batch):
        """
        Apply attention mechanism to aggregate neighbor features, then perform classification
        (Graph pooling by averaging node embeddings)
        :param x: node features
        :param edge_index: graph connectivity
        :param batch: batch indices for graph-level pooling
        :return: graph-level predictions
        """

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)

        x = global_mean_pool(x, batch)

        x = F.dropout(x, p=self.p, training=self.training)
        x = self.lin(x)

        return x
