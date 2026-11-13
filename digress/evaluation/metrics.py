from collections import Counter

import torch
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from digress.data.chem import NUM_ATOM_TYPES, NUM_BOND_TYPES, build_mol, flat_smiles


def evaluate(atoms, bonds, masks, train_smiles_flat):
    n = len(atoms)
    fails, n_valid, n_connected, good = Counter(), 0, 0, []
    for a, b, m in zip(atoms, bonds, masks):
        mol = build_mol(a, b, m)
        connected = len(Chem.GetMolFrags(mol)) == 1
        n_connected += connected
        try:
            Chem.SanitizeMol(mol)
        except Exception as err:
            fails["valence" if "valence" in str(err).lower() else "other"] += 1
            continue
        n_valid += 1
        if connected:
            good.append(flat_smiles(mol))
    unique = set(good)
    n_novel = sum(s not in train_smiles_flat for s in unique)
    return {
        "n": n,
        "validity": n_valid / n,
        "connected": n_connected / n,
        "valid_connected": len(good) / n,
        "uniqueness": len(unique) / len(good) if good else 0.0,
        "novelty": n_novel / len(unique) if unique else 0.0,
        "vun": n_novel / n,
        "fail_valence": fails["valence"] / n,
        "fail_other": fails["other"] / n,
        "bonds_per_mol": (bonds.triu(1) > 0).sum((1, 2)).float().mean().item(),
    }


def _tv(p, q):
    return (
        0.5 * (p / p.sum().clamp(min=1) - q / q.sum().clamp(min=1)).abs().sum().item()
    )


def graph_histograms(atoms, adj, mask):
    atoms, adj, mask = atoms.cpu(), adj.cpu(), mask.cpu().bool()
    max_atoms = atoms.shape[1]
    n_valence = 3 * (max_atoms - 1) + 1
    pairs = mask[:, :, None] & mask[:, None, :]
    valence = (adj * pairs).sum(-1)
    return {
        "n_atoms": torch.bincount(mask.sum(1), minlength=max_atoms + 1).float(),
        "atom_types": torch.bincount(atoms[mask], minlength=NUM_ATOM_TYPES).float(),
        "bond_types": torch.bincount(
            adj[torch.triu(pairs, diagonal=1)], minlength=NUM_BOND_TYPES
        ).float(),
        "valence": torch.bincount(
            (atoms * n_valence + valence)[mask], minlength=NUM_ATOM_TYPES * n_valence
        ).float(),
    }


def distribution_distances(generated, reference):
    return {f"tv_{k}": _tv(generated[k], reference[k]) for k in reference}
