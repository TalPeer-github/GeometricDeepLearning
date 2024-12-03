import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.nn import ChebConv


class BasicChebNet(torch.nn.Module):
    """
    ChebNet is a spectral-based Graph Neural Network that applies Chebyshev polynomials to approximate the graph
    convolution operation. Spectral methods like the Graph Convolution Network (GCN) rely on the eigenvalues of the
    graph Laplacian, which can be computationally expensive for large graphs. ChebNet overcomes this limitation by
    using a truncated expansion of Chebyshev polynomials.

    How it works:
        Spectral convolution: In spectral graph convolution, a graph filter is applied to the graph
            Laplacian's eigenvectors. However, direct computation of the eigenvectors is costly for large graphs.
        Chebyshev approximation: Instead of calculating the eigenvectors, ChebNet approximates the spectral convolution
            by using Chebyshev polynomials. These polynomials are used to approximate the graph convolution operation,
            reducing the computational cost.
        Chebyshev filter: The convolution operation is approximated using a truncated
            expansion of Chebyshev polynomials of the graph Laplacian. The order of the polynomial, denoted by K,
            determines how many terms of the expansion are used.
    """

    def __init__(self, num_node_features, hidden_channels, num_classes, p=0.4, seed=42, k_order=2):
        super(BasicChebNet, self).__init__()
        torch.manual_seed(seed=seed)
        self.p = p

        self.conv1 = ChebConv(num_node_features, hidden_channels, K=k_order)
        self.conv2 = ChebConv(hidden_channels, hidden_channels, K=k_order)
        self.conv3 = ChebConv(hidden_channels, hidden_channels, K=k_order)
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


class ChebNet(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes, p=0.4, seed=42, k_order=2,
                 num_lin=2, num_conv=4, pooling_type="mean", print_params_flag=False):
        super(ChebNet, self).__init__()

        torch.manual_seed(seed)
        self.p = p
        self.k_order = k_order
        self.num_lin = num_lin
        self.num_conv = num_conv
        self.pooling_type = pooling_type

        self.convs = torch.nn.ModuleList(
            [ChebConv(num_node_features if i == 0 else hidden_channels, hidden_channels, K=k_order)
             for i in range(num_conv)]
        )

        lin_sizes = [hidden_channels] + [hidden_channels // (2 ** (i + 1)) for i in range(num_lin - 1)] + [num_classes]
        self.linears = torch.nn.ModuleList(
            [Linear(lin_sizes[i], lin_sizes[i + 1]) for i in range(num_lin)]
        )

        if print_params_flag:
            self.print_params()

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = x.relu()

        if self.pooling_type == "mean":
            x = global_mean_pool(x, batch)
        elif self.pooling_type == "add":
            x = global_add_pool(x, batch)
        elif self.pooling_type == "max":
            x = global_max_pool(x, batch)

        x = F.dropout(x, p=self.p, training=self.training)
        for lin in self.linears[:-1]:
            x = lin(x).relu()
        x = self.linears[-1](x)
        return x

    def print_params(self):
        print("ChebNet ->")
        print(f"\n\tnum_conv_layers = {self.num_conv}"
              f"\n\tnum_linear_layers = {self.num_lin}"
              f"\n\tk_order = {self.k_order}"
              f"\n\tpooling_type = {self.pooling_type}\n")
