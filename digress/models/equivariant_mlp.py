import torch
import torch.nn as nn

from digress.models.conditioning import condition_vector


def _mlp(d_in, d_hidden, d_out):
    return nn.Sequential(nn.Linear(d_in, d_hidden), nn.SiLU(), nn.Linear(d_hidden, d_out))


class EquivariantMLP(nn.Module):
    def __init__(self, num_nodes, num_edges, num_node_classes, num_edge_classes, time_dim, hidden_dim=128, y_dim=0):
        super().__init__()
        self.y_dim, self.time_dim = y_dim, time_dim
        self.register_buffer("triu", torch.triu_indices(num_nodes, num_nodes, offset=1), persistent=False)
        h = hidden_dim
        self.node_enc, self.edge_enc = _mlp(num_node_classes, h, h), _mlp(num_edge_classes, h, h)
        self.cond_enc, self.glob = _mlp(time_dim + y_dim, h, h), _mlp(3 * h, h, h)
        self.node_head, self.edge_head = _mlp(2 * h, h, num_node_classes), _mlp(4 * h, h, num_edge_classes)

    def forward(self, X, E, y, t):
        i, j = self.triu
        node_mask = (X.abs().sum(-1, keepdim=True) > 0).float()
        edge_mask = node_mask[:, i] * node_mask[:, j]
        h, e = self.node_enc(X) * node_mask, self.edge_enc(E) * edge_mask
        g = self.glob(torch.cat([h.sum(1) / node_mask.sum(1).clamp(min=1), e.sum(1) / edge_mask.sum(1).clamp(min=1),
                                 self.cond_enc(condition_vector(t, y, self.time_dim, self.y_dim))], -1))
        hi, hj = h[:, i], h[:, j]
        nodes = self.node_head(torch.cat([h, g[:, None].expand_as(h)], -1))
        edges = self.edge_head(torch.cat([e, hi + hj, hi * hj, g[:, None].expand_as(e)], -1))
        return nodes, edges
