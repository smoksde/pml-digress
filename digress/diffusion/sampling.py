"""
changes: shapes come from info now, sampling in batches (SAMPLE_BATCH_SIZE).
z_T from each own diffusion's limit dist. for supporting also uniform alongside marginal.
"""

import torch
import torch.nn.functional as F

from digress.config import DEVICE, DIFFUSION_STEPS, MAX_ATOMS, NUM_EDGES
from digress.data.chem import NUM_ATOM_TYPES, NUM_BOND_TYPES
from digress.data.graphs import edge_mask_from_nodes


def _init_xt(marginals, num_samples):
    counts = torch.multinomial(marginals["atom_count_dist"], num_samples, replacement=True).to(DEVICE)
    mask = torch.arange(MAX_ATOMS, device=DEVICE).unsqueeze(0) < counts.unsqueeze(1)
    edge_mask = edge_mask_from_nodes(mask)
    atoms_t = torch.multinomial(marginals["atom_marginal"], num_samples * MAX_ATOMS, replacement=True)
    atoms_t = atoms_t.view(num_samples, MAX_ATOMS).to(DEVICE) * mask
    bonds_t = torch.multinomial(marginals["bond_marginal"], num_samples * NUM_EDGES, replacement=True)
    bonds_t = bonds_t.view(num_samples, NUM_EDGES).to(DEVICE) * edge_mask
    return atoms_t, bonds_t, mask, edge_mask


@torch.no_grad()
def generate(model, atom_diffusion, bond_diffusion, marginals, num_samples, y_in=None):
    atoms_t, bonds_t, mask, edge_mask = _init_xt(marginals, num_samples)
    for t in range(DIFFUSION_STEPS, 0, -1):
        t_batch = torch.full((num_samples,), t, device=DEVICE, dtype=torch.long)
        X = F.one_hot(atoms_t, NUM_ATOM_TYPES).float() * mask.unsqueeze(-1)  # padding = zero rows
        E = F.one_hot(bonds_t, NUM_BOND_TYPES).float() * edge_mask.unsqueeze(-1)
        atom_logits, bond_logits = model(X, E, y_in, t_batch)
        atoms_t = atom_diffusion.reverse_step(atoms_t, atom_logits, t) * mask
        bonds_t = bond_diffusion.reverse_step(bonds_t, bond_logits, t) * edge_mask
    return atoms_t, bonds_t, mask, edge_mask
