import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import degree
from GCN import GCN


class EnergyAwareGCN(GCN):
    def __init__(self, num_node_features, hidden_channels, num_classes, p=0.4, seed=42):
        super(EnergyAwareGCN, self).__init__(num_node_features, hidden_channels, num_classes, p, seed)
        self.energy_factor = torch.nn.Parameter(torch.Tensor([1.0]))

    def forward(self, x, edge_index, batch, energy=None):
        """
        Incorporates an energy feature (optional) in the message-passing procedure.
        :param energy: Node-specific energy information.
        """
        x = self.conv1(x, edge_index)
        if energy is not None:
            x = x * (self.energy_factor * energy.view(-1, 1))
        x = x.relu()

        x = self.conv2(x, edge_index)
        if energy is not None:
            x = x * (self.energy_factor * energy.view(-1, 1))
        x = x.relu()

        x = self.conv3(x, edge_index)
        if energy is not None:
            x = x * (self.energy_factor * energy.view(-1, 1))

        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=self.p, training=self.training)
        x = self.lin(x)

        return x


class OrbitalAwareGCN(EnergyAwareGCN):
    def __init__(self, num_node_features, hidden_channels, num_classes, p=0.4, seed=42):
        super(OrbitalAwareGCN, self).__init__(num_node_features, hidden_channels, num_classes, p, seed)
        self.atom_orbitals = None

    def set_orbitals(self, orbitals):
        """
        Setting atomic orbital features (need to set quant chemistry methods)
        """
        self.atom_orbitals = orbitals

    def forward(self, x, edge_index, batch):
        """
        Modifies the GCN to consider orbital information by adjusting edge weights during MP.
        :param x: Node features
        :param edge_index: Graph connectivity
        :param batch: Batch index
        :return: Output predictions
        """
        if self.atom_orbitals is not None:

            edge_weights = self.atom_orbitals[edge_index[0], edge_index[1]]

            x = self.conv1(x, edge_index, edge_weight=edge_weights)
        else:
            x = self.conv1(x, edge_index)

        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=self.p, training=self.training)
        x = self.lin(x)

        return x


class ChargeAwareGCN(OrbitalAwareGCN):
    def __init__(self, num_node_features, hidden_channels, num_classes, p=0.4, seed=42):
        super(ChargeAwareGCN, self).__init__(num_node_features, hidden_channels, num_classes, p, seed)
        self.electronegativity_factor = torch.nn.Parameter(torch.Tensor([1.0]))  # Electronegativity influence

    def forward(self, x, edge_index, batch, electronegativity=None):
        if electronegativity is not None:
            x = x * (self.electronegativity_factor * electronegativity.view(-1, 1))

        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=self.p, training=self.training)
        x = self.lin(x)

        return x
