import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import GINConv
from torch_geometric.nn import BatchNorm


class GIN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes, p=0.4, seed=42):
        super(GIN, self).__init__()
        torch.manual_seed(seed=seed)
        self.p = p

        nn1 = torch.nn.Sequential(
            torch.nn.Linear(num_node_features, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels),
        )
        self.conv1 = GINConv(nn1)
        self.conv2 = GINConv(nn1)
        self.conv3 = GINConv(nn1)

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
