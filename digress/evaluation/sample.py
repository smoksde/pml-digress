"""
changes: use load_run() and info
"""

import torch

from digress.config import DEVICE
from digress.data.graphs import edge_vector_to_adj
from digress.diffusion.sampling import generate
from digress.models.conditioning import normalize
from digress.training.checkpoint import load_run


@torch.no_grad()
def sample(num_samples, checkpoint, y=None):
    ckpt, run = load_run(checkpoint)
    y_in = None
    if y is not None:
        if not run.model.y_dim:
            raise ValueError("this checkpoint is unconditional; train with LABEL_PROPERTY set to use y")
        y_in = normalize(torch.full((num_samples, run.model.y_dim), float(y), device=DEVICE), ckpt["y_mean"],
                         ckpt["y_std"])
    atoms, bonds, mask, _ = generate(run.model, run.atom_diffusion, run.bond_diffusion,
                                     run.marginals["atom_count_dist"], num_samples, run.info, y_in)
    return atoms, edge_vector_to_adj(bonds, run.info.max_atoms), mask, run
