import torch
import torch.nn.functional as F

from digress.config import DEVICE, SAMPLE_BATCH_SIZE
from digress.data.graphs import edge_mask_from_nodes


def _init_xt(atom_dist, bond_dist, count_dist, num_samples, info):
    n, n_edges = info.max_atoms, info.num_edges
    counts = torch.multinomial(count_dist.float(), num_samples, replacement=True).to(DEVICE)
    mask = torch.arange(n, device=DEVICE).unsqueeze(0) < counts.unsqueeze(1)
    edge_mask = edge_mask_from_nodes(mask)
    atoms_t = torch.multinomial(atom_dist.float(), num_samples * n, replacement=True)
    atoms_t = atoms_t.view(num_samples, n).to(DEVICE) * mask
    bonds_t = torch.multinomial(bond_dist.float(), num_samples * n_edges, replacement=True)
    bonds_t = bonds_t.view(num_samples, n_edges).to(DEVICE) * edge_mask
    return atoms_t, bonds_t, mask, edge_mask


@torch.no_grad()
def _generate_batch(model, atom_diffusion, bond_diffusion, count_dist, num_samples, info, y_in):
    # start from the noise model's own limit distribution (marginal / uniform) and not data marginals
    atoms_t, bonds_t, mask, edge_mask = _init_xt(atom_diffusion.m, bond_diffusion.m, count_dist, num_samples, info)
    for t in range(atom_diffusion.T, 0, -1):
        t_batch = torch.full((num_samples,), t, device=DEVICE, dtype=torch.long)
        X = F.one_hot(atoms_t, info.num_atom_types).float() * mask.unsqueeze(-1)  # padding = zero rows
        E = F.one_hot(bonds_t, info.num_bond_types).float() * edge_mask.unsqueeze(-1)
        atom_logits, bond_logits = model(X, E, y_in, t_batch)
        atoms_t = atom_diffusion.reverse_step(atoms_t, atom_logits, t) * mask
        bonds_t = bond_diffusion.reverse_step(bonds_t, bond_logits, t) * edge_mask
    return atoms_t, bonds_t, mask, edge_mask


@torch.no_grad()
def generate(model, atom_diffusion, bond_diffusion, count_dist, num_samples, info, y_in=None,
             batch_size=SAMPLE_BATCH_SIZE):
    out = []
    for start in range(0, num_samples, batch_size):
        n = min(batch_size, num_samples - start)
        y = None if y_in is None else y_in[start : start + n]
        out.append(_generate_batch(model, atom_diffusion, bond_diffusion, count_dist, n, info, y))
    return tuple(torch.cat(parts) for parts in zip(*out))
