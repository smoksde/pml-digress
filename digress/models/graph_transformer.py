import torch
import torch.nn as nn

from digress.models.conditioning import condition_vector


class GraphLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.Q, self.K, self.V, self.E = (nn.Linear(dim, dim) for _ in range(4))
        self.O_h, self.O_e = nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.norm_h, self.norm_e = nn.LayerNorm(dim), nn.LayerNorm(dim)

    def forward(self, h, e, mask):
        q, k, v = self.Q(h), self.K(h), self.V(h)
        scores = q.unsqueeze(2) * k.unsqueeze(1) / q.shape[-1] ** 0.5 * self.E(e)
        attn = (
            scores.sum(-1).masked_fill(~mask[:, None, :], float("-inf")).softmax(dim=2)
        )
        h = self.norm_h(h + self.O_h(attn @ v))
        e = self.norm_e(e + self.O_e(scores))
        return h, e


class GraphTransformer(nn.Module):
    def __init__(
        self,
        num_nodes,
        num_edges,
        num_node_classes,
        num_edge_classes,
        time_dim,
        hidden_dim=128,
        y_dim=0,
        num_layers=4,
    ):
        super().__init__()
        self.num_nodes, self.y_dim, self.time_dim = num_nodes, y_dim, time_dim
        self.register_buffer(
            "triu", torch.triu_indices(num_nodes, num_nodes, offset=1), persistent=False
        )
        self.node_in = nn.Linear(num_node_classes, hidden_dim)
        self.edge_in = nn.Linear(num_edge_classes, hidden_dim)
        self.cond_in = nn.Linear(time_dim + y_dim, hidden_dim)
        self.layers = nn.ModuleList(GraphLayer(hidden_dim) for _ in range(num_layers))
        self.node_out = nn.Linear(hidden_dim, num_node_classes)
        self.edge_out = nn.Linear(hidden_dim, num_edge_classes)

    def forward(self, X, E, y, t):
        i, j = self.triu
        mask = X.abs().sum(-1) > 0
        E_mat = E.new_zeros(E.shape[0], self.num_nodes, self.num_nodes, E.shape[-1])
        E_mat[:, i, j] = E
        h = (
            self.node_in(X)
            + self.cond_in(condition_vector(t, y, self.time_dim, self.y_dim))[:, None]
        )
        e = self.edge_in(E_mat + E_mat.transpose(1, 2))
        for layer in self.layers:
            h, e = layer(h, e, mask)
        return self.node_out(h), self.edge_out(((e + e.transpose(1, 2)) / 2)[:, i, j])
