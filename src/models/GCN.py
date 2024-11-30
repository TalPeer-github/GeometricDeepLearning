import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import degree


class GCN(torch.nn.Module):
    """
    GCN with ReLU(x) = max(x, 0) activation for obtaining localized node embeddings,
    before we apply our final classifier on top of a graph readout layer.
    """

    def __init__(self, num_node_features, hidden_channels, num_classes, p=0.4, seed=42):
        super(GCN, self).__init__()
        torch.manual_seed(seed=seed)
        self.p = p

        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        """
        Obtain node embeddings, add readout layer, and apply a final classifier
        :param x: node
        :param edge_index:
        :param batch:
        :return:
        """

        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        x = F.dropout(x, p=self.p, training=self.training)
        x = self.lin(x)

        return x

