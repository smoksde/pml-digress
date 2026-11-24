"""Graph Transformer of DiGress authors, based on the reference implementation (/src/models/transformer_model.py and /src/models/layers.py)

Nodes, edges and the global vector y are updated together. Attention scores are computed per node pair and modified by
the edge features (FiLM layer), the scores become the new edge features and y modifies both nodes and edges

Without using node-index features every operation is permutation equivariant and the outputs permute with the inputs

FOR NOW: NO STRUCTURAL/ SPECTRAL FEATURES

"""
import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from digress.features import ExtraFeatures, FeatureConfig


class Xtoy(nn.Module):
    """Map node features to global features (mean, min, max, std)"""

    def __init__(self, dx, dy):
        super().__init__()
        self.lin = nn.Linear(4 * dx, dy)

    def forward(self, X):
        z = torch.hstack((X.mean(dim=1), X.min(dim=1)[0], X.max(dim=1)[0], X.std(dim=1)))
        return self.lin(z)


class Etoy(nn.Module):
    """Map edge features to global features"""

    def __init__(self, d, dy):
        super().__init__()
        self.lin = nn.Linear(4 * d, dy)

    def forward(self, E):
        z = torch.hstack((E.mean(dim=(1, 2)), E.min(dim=2)[0].min(dim=1)[0], E.max(dim=2)[0].max(dim=1)[0],
                          torch.std(E, dim=(1, 2))))
        return self.lin(z)


def masked_softmax(x, mask, **kwargs):
    if mask.sum() == 0:
        return x
    x_masked = x.clone()
    x_masked[mask == 0] = -float("inf")
    return torch.softmax(x_masked, **kwargs)


def _mask(X, E, node_mask):
    x_mask = node_mask.unsqueeze(-1).type_as(X)
    e_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(1)
    return X * x_mask, E * e_mask


class NodeEdgeBlock(nn.Module):
    """Self-attention layer that also updates the edge representations"""

    def __init__(self, dx, de, dy, n_head):
        super().__init__()
        assert dx % n_head == 0, f"dx: {dx} -- nhead: {n_head}"
        self.dx, self.de, self.dy, self.n_head = dx, de, dy, n_head
        self.df = dx // n_head

        # Attention
        self.q, self.k, self.v = nn.Linear(dx, dx), nn.Linear(dx, dx), nn.Linear(dx, dx)

        # FiLM E -> X
        self.e_add, self.e_mul = nn.Linear(de, dx), nn.Linear(de, dx)

        # FiLM y -> E (here dx and not de as in the authors' code)
        self.y_e_mul, self.y_e_add = nn.Linear(dy, dx), nn.Linear(dy, dx)

        # FiLM y -> X
        self.y_x_mul, self.y_x_add = nn.Linear(dy, dx), nn.Linear(dy, dx)

        # Processing y
        self.y_y = nn.Linear(dy, dy)
        self.x_y = Xtoy(dx, dy)
        self.e_y = Etoy(de, dy)

        # outputs
        self.x_out = nn.Linear(dx, dx)
        self.e_out = nn.Linear(dx, de)
        self.y_out = nn.Sequential(nn.Linear(dy, dy), nn.ReLU(), nn.Linear(dy, dy))

    def forward(self, X, E, y, node_mask):
        bs, n, _ = X.shape
        x_mask = node_mask.unsqueeze(-1)  # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)     # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)     # bs, 1, n, 1

        # Mapping X to keys and queries
        Q = (self.q(X) * x_mask).reshape(bs, n, self.n_head, self.df).unsqueeze(2)  # bs, n, 1, h, df
        K = (self.k(X) * x_mask).reshape(bs, n, self.n_head, self.df).unsqueeze(1)  # bs, 1, n, h, df
        Y = Q * K / math.sqrt(self.df)  # bs, n, n, h, df

        E1 = (self.e_mul(E) * e_mask1 * e_mask2).reshape(bs, n, n, self.n_head, self.df)
        E2 = (self.e_add(E) * e_mask1 * e_mask2).reshape(bs, n, n, self.n_head, self.df)
        Y = Y * (E1 + 1) + E2

        # edges: incorporating y to E then FiLM by y
        newE = Y.flatten(start_dim=3)  # bs, n, n, dx
        ye1 = self.y_e_add(y).unsqueeze(1).unsqueeze(1)
        ye2 = self.y_e_mul(y).unsqueeze(1).unsqueeze(1)
        newE = ye1 + (ye2 + 1) * newE
        newE = self.e_out(newE) * e_mask1 * e_mask2  # bs, n, n, de

        # nodes: attention over the keys of real nodes
        softmax_mask = e_mask2.expand(-1, n, -1, self.n_head)
        attn = masked_softmax(Y, softmax_mask, dim=2)
        V = (self.v(X) * x_mask).reshape(bs, n, self.n_head, self.df).unsqueeze(1)  # bs, 1, n, h, df
        weighted_V = (attn * V).sum(dim=2).flatten(start_dim=2)  # bs, n, dx
        
        # Incorporating y to X
        yx1 = self.y_x_add(y).unsqueeze(1)
        yx2 = self.y_x_mul(y).unsqueeze(1)
        newX = self.x_out(yx1 + (yx2 + 1) * weighted_V) * x_mask

        # global
        new_y = self.y_out(self.y_y(y) + self.x_y(X) + self.e_y(E))
        return newX, newE, new_y


class XEyTransformerLayer(nn.Module):
    def __init__(self, dx, de, dy, n_head, dim_ffX=2048, dim_ffE=128, dim_ffy=2048, dropout=0.1, layer_norm_eps=1e-5):
        super().__init__()
        self.self_attn = NodeEdgeBlock(dx, de, dy, n_head)

        self.linX1, self.linX2 = nn.Linear(dx, dim_ffX), nn.Linear(dim_ffX, dx)
        self.normX1, self.normX2 = nn.LayerNorm(dx, eps=layer_norm_eps), nn.LayerNorm(dx, eps=layer_norm_eps)
        self.dropoutX1, self.dropoutX2, self.dropoutX3 = (nn.Dropout(dropout) for _ in range(3))

        self.linE1, self.linE2 = nn.Linear(de, dim_ffE), nn.Linear(dim_ffE, de)
        self.normE1, self.normE2 = nn.LayerNorm(de, eps=layer_norm_eps), nn.LayerNorm(de, eps=layer_norm_eps)
        self.dropoutE1, self.dropoutE2, self.dropoutE3 = (nn.Dropout(dropout) for _ in range(3))

        self.lin_y1, self.lin_y2 = nn.Linear(dy, dim_ffy), nn.Linear(dim_ffy, dy)
        self.norm_y1, self.norm_y2 = nn.LayerNorm(dy, eps=layer_norm_eps), nn.LayerNorm(dy, eps=layer_norm_eps)
        self.dropout_y1, self.dropout_y2, self.dropout_y3 = (nn.Dropout(dropout) for _ in range(3))

        self.activation = F.relu

    def forward(self, X, E, y, node_mask):
        newX, newE, new_y = self.self_attn(X, E, y, node_mask=node_mask)

        X = self.normX1(X + self.dropoutX1(newX))
        E = self.normE1(E + self.dropoutE1(newE))
        y = self.norm_y1(y + self.dropout_y1(new_y))

        X = self.normX2(X + self.dropoutX3(self.linX2(self.dropoutX2(self.activation(self.linX1(X))))))
        E = self.normE2(E + self.dropoutE3(self.linE2(self.dropoutE2(self.activation(self.linE1(E))))))
        y = self.norm_y2(y + self.dropout_y3(self.lin_y2(self.dropout_y2(self.activation(self.lin_y1(y))))))
        return X, E, y


class GraphTransformer(nn.Module):
    """Dense DiGress net: X (bs, n, dx_in), E (bs, n, n, de_in), y (bs, dy_in) -> logits of the (clean) graph"""


    def __init__(self, n_layers, input_dims, hidden_mlp_dims, hidden_dims, output_dims, dropout=0.1):
        super().__init__()
        act_in, act_out = nn.ReLU(), nn.ReLU()
        self.out_dim_X, self.out_dim_E, self.out_dim_y = output_dims["X"], output_dims["E"], output_dims["y"]

        self.mlp_in_X = nn.Sequential(nn.Linear(input_dims["X"], hidden_mlp_dims["X"]), act_in,
                                      nn.Linear(hidden_mlp_dims["X"], hidden_dims["dx"]), act_in)
        self.mlp_in_E = nn.Sequential(nn.Linear(input_dims["E"], hidden_mlp_dims["E"]), act_in,
                                      nn.Linear(hidden_mlp_dims["E"], hidden_dims["de"]), act_in)
        self.mlp_in_y = nn.Sequential(nn.Linear(input_dims["y"], hidden_mlp_dims["y"]), act_in,
                                      nn.Linear(hidden_mlp_dims["y"], hidden_dims["dy"]), act_in)

        self.tf_layers = nn.ModuleList(
            XEyTransformerLayer(dx=hidden_dims["dx"], de=hidden_dims["de"], dy=hidden_dims["dy"],
                                n_head=hidden_dims["n_head"], dim_ffX=hidden_dims["dim_ffX"],
                                dim_ffE=hidden_dims["dim_ffE"], dim_ffy=hidden_dims["dim_ffy"], dropout=dropout)
            for _ in range(n_layers))

        self.mlp_out_X = nn.Sequential(nn.Linear(hidden_dims["dx"], hidden_mlp_dims["X"]), act_out,
                                       nn.Linear(hidden_mlp_dims["X"], output_dims["X"]))
        self.mlp_out_E = nn.Sequential(nn.Linear(hidden_dims["de"], hidden_mlp_dims["E"]), act_out,
                                       nn.Linear(hidden_mlp_dims["E"], output_dims["E"]))
        # no global prediction target in unconditional generation setup: skipping the y head (would be width 0)
        self.mlp_out_y = (nn.Sequential(nn.Linear(hidden_dims["dy"], hidden_mlp_dims["y"]), act_out,
                                        nn.Linear(hidden_mlp_dims["y"], output_dims["y"]))
                          if output_dims["y"] else None) #None

    def forward(self, X, E, y, node_mask):
        bs, n = X.shape[0], X.shape[1]
        diag_mask = ~torch.eye(n, dtype=torch.bool, device=X.device)
        diag_mask = diag_mask.unsqueeze(0).unsqueeze(-1).expand(bs, -1, -1, -1)

        X_to_out = X[..., : self.out_dim_X]
        E_to_out = E[..., : self.out_dim_E]
        y_to_out = y[..., : self.out_dim_y]

        new_E = self.mlp_in_E(E)
        new_E = (new_E + new_E.transpose(1, 2)) / 2
        X, E = _mask(self.mlp_in_X(X), new_E, node_mask)
        y = self.mlp_in_y(y)

        for layer in self.tf_layers:
            X, E, y = layer(X, E, y, node_mask)

        X = self.mlp_out_X(X) + X_to_out
        E = (self.mlp_out_E(E) + E_to_out) * diag_mask
        y = self.mlp_out_y(y) + y_to_out if self.mlp_out_y is not None else y_to_out
        E = 0.5 * (E + E.transpose(1, 2))
        X, E = _mask(X, E, node_mask)
        return X, E, y


class DiGressTransformer(nn.Module):
    """
    Utility: keep interface (X, E_vec, y, t) building dense, symmetric adj., appends extra features, builds global input 
    y = [t/T, extra_y (extra global features), conditioning label]
    """

    needs_info = True

    def __init__(self, num_nodes, num_edges, num_node_classes, num_edge_classes, *, info, diffusion_steps, n_layers,
                 hidden_mlp_dims, hidden_dims, dropout=0.1, y_dim=0, features=None):
        super().__init__()
        self.num_nodes, self.y_dim, self.T = num_nodes, y_dim, diffusion_steps
        self.register_buffer("triu", torch.triu_indices(num_nodes, num_nodes, offset=1), persistent=False)
        self.features = ExtraFeatures(FeatureConfig.from_dict(features), info)
        fx, fe, fy = self.features.dims
        self.net = GraphTransformer(
            n_layers=n_layers,
            input_dims={"X": num_node_classes + fx, "E": num_edge_classes + fe, "y": 1 + fy + y_dim},
            hidden_mlp_dims=hidden_mlp_dims, hidden_dims=hidden_dims,
            output_dims={"X": num_node_classes, "E": num_edge_classes, "y": 0}, dropout=dropout)

    def forward(self, X, E, y, t):
        i, j = self.triu
        bs, n = X.shape[0], self.num_nodes
        node_mask = X.abs().sum(-1) > 0
        E_dense = E.new_zeros(bs, n, n, E.shape[-1])
        E_dense[:, i, j] = E
        E_dense = E_dense + E_dense.transpose(1, 2)

        x_extra, e_extra, y_extra = self.features(X, E_dense, node_mask)
        parts = [(t.float() / self.T).unsqueeze(-1), y_extra]
        if self.y_dim:
            if y is None:
                raise ValueError("this model is conditional; pass a label y")
            parts.append(y)
        X_out, E_out, _ = self.net(torch.cat([X, x_extra], -1), torch.cat([E_dense, e_extra], -1),
                                   torch.cat(parts, -1), node_mask)
        return X_out, E_out[:, i, j]
