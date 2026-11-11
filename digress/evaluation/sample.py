import torch

from digress.config import DEVICE, MAX_ATOMS
from digress.data.graphs import edge_vector_to_adj
from digress.diffusion.sampling import generate
from digress.models.conditioning import normalize
from digress.training.checkpoint import load_for_sampling


@torch.no_grad()
def sample(num_samples, checkpoint, y=None):
    ckpt, model, atom_diffusion, bond_diffusion = load_for_sampling(checkpoint)
    y_in = None
    if y is not None:
        if not model.y_dim:
            raise ValueError(
                "this checkpoint is unconditional; train with LABEL_PROPERTY set to use y"
            )
        y_in = normalize(
            torch.full((num_samples, model.y_dim), float(y), device=DEVICE),
            ckpt["y_mean"],
            ckpt["y_std"],
        )
    atoms, bonds, mask, _ = generate(
        model, atom_diffusion, bond_diffusion, ckpt, num_samples, y_in
    )
    return atoms, edge_vector_to_adj(bonds, MAX_ATOMS), mask
