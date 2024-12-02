import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import EGConv


class EGCNet(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes, p=0.4, seed=42):
        super(EGCNet, self).__init__()

        torch.manual_seed(seed=seed)
        self.p = p

        self.conv1 = EGConv(num_node_features, hidden_channels)
        self.conv2 = EGConv(hidden_channels, hidden_channels)
        self.conv3 = EGConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        x = F.dropout(x, p=self.p, training=self.training)
        x = self.lin(x)

        return x
