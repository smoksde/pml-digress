import torch.nn.functional as F

from digress.config import DEVICE
from digress.data.chem import NUM_ATOM_TYPES, NUM_BOND_TYPES
from digress.data.dataset import get_dataset
from digress.data.graphs import edge_mask_from_nodes
from digress.diffusion.discrete import noise_graph
from digress.models.conditioning import normalize


def _masked_mean(values, mask):
    mask = mask.float()
    return (values * mask).sum() / mask.sum().clamp(min=1)


def compute_loss(atom_logits, bond_logits, atoms0, bonds0, node_mask, edge_mask, edge_weight=1.0):
    atom_ce = F.cross_entropy(atom_logits.flatten(0, 1), atoms0.flatten(), reduction="none")
    bond_ce = F.cross_entropy(bond_logits.flatten(0, 1), bonds0.flatten(), reduction="none")
    atom_loss = _masked_mean(atom_ce.view_as(atoms0), node_mask)
    bond_loss = _masked_mean(bond_ce.view_as(bonds0), edge_mask)
    return atom_loss + edge_weight * bond_loss, atom_loss, bond_loss


def denoise_loss(run, batch, t, edge_weight):
    atoms0, mask0, bonds0, y0 = (None if v is None else v.to(DEVICE) for v in batch)
    X0 = F.one_hot(atoms0, NUM_ATOM_TYPES).float()
    E0 = F.one_hot(bonds0, NUM_BOND_TYPES).float()
    X, E = noise_graph(X0, E0, mask0, t, run.atom_diffusion, run.bond_diffusion)
    y_in = None
    if run.model.y_dim:
        data = get_dataset()
        y_in = normalize(y0, data.y_mean, data.y_std)
    atom_logits, bond_logits = run.model(X, E, y_in, t)
    return compute_loss(atom_logits, bond_logits, atoms0, bonds0, mask0, edge_mask_from_nodes(mask0), edge_weight)
