import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import DiffPool


class DiffPoolNet(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes, p=0.4, seed=42):
        super(DiffPoolNet, self).__init__()
        torch.manual_seed(seed=seed)
        self.p = p

        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.pool1 = DiffPool(hidden_channels, hidden_channels, num_classes)
        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()

        x, edge_index, batch, _, _ = self.pool1(x, edge_index, batch)

        x = F.dropout(x, p=self.p, training=self.training)
        x = self.lin(x)

        return x
