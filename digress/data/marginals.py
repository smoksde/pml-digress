import torch

from digress.config import MAX_ATOMS, MARGINAL_SAMPLE_SIZE
from digress.data.dataset import get_dataset
from digress.data.graphs import edge_mask_from_nodes


def _empirical_marginal(num_classes, samples):
    counts = torch.bincount(samples.reshape(-1), minlength=num_classes).float()
    return counts / counts.sum()


def bootstrap_stats(num_atom_types, num_bond_types):
    data = get_dataset()
    idx = data.train_idx[:MARGINAL_SAMPLE_SIZE]
    atoms, masks, bonds = data.atoms[idx], data.masks[idx], data.edge_vecs[idx]
    atom_marginal = _empirical_marginal(num_atom_types, atoms[masks.bool()])
    bond_marginal = _empirical_marginal(num_bond_types, bonds[edge_mask_from_nodes(masks.bool())])  # real edges only
    counts = masks.sum(dim=1)
    atom_count_dist = torch.bincount(counts, minlength=MAX_ATOMS + 1).float()
    return atom_marginal, bond_marginal, atom_count_dist / atom_count_dist.sum()
