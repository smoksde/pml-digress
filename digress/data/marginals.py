"""
smaller chganges, incorporating the new subsample() fcuntion
"""

import torch

from digress.config import MARGINAL_SAMPLE_SIZE
from digress.data.dataset import subsample
from digress.data.graphs import edge_mask_from_nodes


def _empirical_marginal(num_classes, samples):
    counts = torch.bincount(samples.reshape(-1).long(), minlength=num_classes).float()
    return counts / counts.sum()


def bootstrap_stats(data, sample_size=MARGINAL_SAMPLE_SIZE):
    info = data.info
    idx = subsample(data.train_idx, sample_size)
    atoms, masks, bonds = data.atoms[idx], data.masks[idx], data.edge_vecs[idx]
    atom_marginal = _empirical_marginal(info.num_atom_types, atoms[masks.bool()])
    bond_marginal = _empirical_marginal(info.num_bond_types, bonds[edge_mask_from_nodes(masks.bool())])
    counts = masks.sum(dim=1)
    atom_count_dist = torch.bincount(counts, minlength=info.max_atoms + 1).float()
    return atom_marginal, bond_marginal, atom_count_dist / atom_count_dist.sum()
