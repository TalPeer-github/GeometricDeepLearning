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


class EnergyAwareGCN(GCN):
    def __init__(self, num_node_features, hidden_channels, num_classes, p=0.4, seed=42):
        super(EnergyAwareGCN, self).__init__(num_node_features, hidden_channels, num_classes, p, seed)
        self.energy_factor = torch.nn.Parameter(torch.Tensor([1.0]))  # Learnable energy factor

    def forward(self, x, edge_index, batch, energy=None):
        """
        Incorporates an energy feature (optional) in the message-passing procedure.
        :param energy: Node-specific energy information (optional).
        """
        x = self.conv1(x, edge_index)
        if energy is not None:
            # Scale node features based on energy information
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
        self.atom_orbitals = None  # Placeholder for atomic orbital features (precomputed or input)

    def set_orbitals(self, orbitals):
        """ Set precomputed atomic orbital features (could be preprocessed from quantum chemistry methods) """
        self.atom_orbitals = orbitals

    def forward(self, x, edge_index, batch):
        """
        Modifies the GCN to consider orbital information by adjusting edge weights during message passing.
        :param x: Node features
        :param edge_index: Graph connectivity
        :param batch: Batch index
        :return: Output predictions
        """
        if self.atom_orbitals is not None:
            # Here we assume the orbital information is associated with edges, adjusting weights.
            edge_weights = self.atom_orbitals[edge_index[0], edge_index[1]]  # Example of using orbital info

            # Use orbital information to scale edge weights during message passing
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
        """
        Modify the GCN by considering electronegativity (or charge) as additional information.
        :param electronegativity: Node-level electronegativity feature
        """
        if electronegativity is not None:
            # Scale node features based on electronegativity information
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
