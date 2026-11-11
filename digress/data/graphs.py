import torch


def adj_to_edge_vector(adj):
    idx = torch.triu_indices(adj.shape[-1], adj.shape[-1], offset=1)
    return adj[..., idx[0], idx[1]]


def edge_mask_from_nodes(mask):
    node_outer = mask[:, :, None] & mask[:, None, :]
    return adj_to_edge_vector(node_outer.long()).bool()


def edge_vector_to_adj(edge_vec, n_nodes):
    idx = torch.triu_indices(n_nodes, n_nodes, offset=1)
    adj = torch.zeros(edge_vec.shape[0], n_nodes, n_nodes, dtype=edge_vec.dtype, device=edge_vec.device)
    adj[:, idx[0], idx[1]] = edge_vec
    return adj + adj.transpose(1, 2)
