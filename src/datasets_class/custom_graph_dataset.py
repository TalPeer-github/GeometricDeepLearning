from torch_geometric.data import InMemoryDataset
import torch_geometric.data


class CustomGraphDataset(InMemoryDataset):
    def __init__(self, data_list):
        """
        Store the list of Graphs Data objects
        :param data_list:
        """
        self.data_list = data_list
        super(CustomGraphDataset, self).__init__(root=None)

    def len(self):
        """
        :return: number of graphs in the dataset
        """
        return len(self.data_list)

    def get(self, idx):
        """
        :param idx: index of interest
        :return: spesific graph at the given index
        """
        return self.data_list[idx]

#
# class Graph(nx.Graph):
#     """
#     edge_index = (2,N): This indicates that there are N edges in the graph.
#         The first dimension (2) means that each edge is represented by two numbers:
#         1. The source node index and the target node index.
#         2. The second dimension (N) indicates the number of edges in the graph.
#     x = [M, 7]: This represents the feature matrix for the nodes in the graph.
#         1. first dimension (M) indicates the number of nodes in the graph,
#         2. second dimension (7) indicates that each node has 7 features (attributes).
#         So, M varies for each graph in your dataset, indicating the number of nodes in each graph,
#         and each of those nodes has 7 features.
#     edge_attr = [N, 4]: This represents the attributes for each edge in the graph. For each edge, there are 4 features.
#         1. first dimension (N) corresponds to the number of edges
#         2. second dimension (4) corresponds to the number of features for each edge.
#     y = [k]: This is the label or target for the graph.
#             The shape [k] means there is a k labels associated with the entire graph.
#     """
#
#     def __init__(self, graph_object):
#         super().__init__(graph_object)
#         #self.label = graph_object.y.numpy().item()
#
