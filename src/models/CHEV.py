import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import ChebConv


class ChebNet(torch.nn.Module):
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

    def __init__(self, num_node_features, hidden_channels, num_classes, p=0.4, seed=42):
        super(ChebNet, self).__init__()
        torch.manual_seed(seed=seed)
        self.p = p

        self.conv1 = ChebConv(num_node_features, hidden_channels, K=2)  # K is the order of the polynomial
        self.conv2 = ChebConv(hidden_channels, hidden_channels, K=2)
        self.conv3 = ChebConv(hidden_channels, hidden_channels, K=2)
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
