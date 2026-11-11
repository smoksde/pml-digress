import torch
import torch.nn as nn

from digress.models.conditioning import condition_vector


class MLP(nn.Module):
    def __init__(self, num_nodes, num_edges, num_node_classes, num_edge_classes, hidden_dim, time_dim, y_dim=0):
        super().__init__()
        self.y_dim, self.time_dim = y_dim, time_dim
        self.node_shape, self.edge_shape = (num_nodes, num_node_classes), (num_edges, num_edge_classes)
        in_dim = num_nodes * num_node_classes + num_edges * num_edge_classes + time_dim + y_dim
        self.trunk = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(),
                                   nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.node_head = nn.Linear(hidden_dim, num_nodes * num_node_classes)
        self.edge_head = nn.Linear(hidden_dim, num_edges * num_edge_classes)

    def forward(self, X, E, y, t):
        b = X.shape[0]
        c = condition_vector(t, y, self.time_dim, self.y_dim)
        h = self.trunk(torch.cat([X.reshape(b, -1), E.reshape(b, -1), c], -1))
        return self.node_head(h).view(b, *self.node_shape), self.edge_head(h).view(b, *self.edge_shape)
