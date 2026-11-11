from collections import Counter

import torch
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from digress.data.chem import build_mol, flat_smiles


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
    return {
        "n": n,
        "validity": n_valid / n,
        "connected": n_connected / n,
        "valid_connected": len(good) / n,
        "uniqueness": len(unique) / len(good) if good else 0.0,
        "novelty": (
            sum(s not in train_smiles_flat for s in unique) / len(unique)
            if unique
            else 0.0
        ),
        "fail_valence": fails["valence"] / n,
        "fail_other": fails["other"] / n,
        "bonds_per_mol": (bonds.triu(1) > 0).sum((1, 2)).float().mean().item(),
    }


def _tv(p, q):
    return 0.5 * (p / p.sum() - q / q.sum()).abs().sum().item()


def distribution_distances(atoms, edge_vecs, mask, edge_mask, marginals):
    atom_hist = torch.bincount(
        atoms[mask].cpu(), minlength=len(marginals["atom_marginal"])
    ).float()
    bond_hist = torch.bincount(
        edge_vecs[edge_mask].cpu(), minlength=len(marginals["bond_marginal"])
    ).float()
    return {
        "tv_atom_types": _tv(atom_hist, marginals["atom_marginal"].cpu()),
        "tv_bond_types": _tv(bond_hist, marginals["bond_marginal"].cpu()),
    }
