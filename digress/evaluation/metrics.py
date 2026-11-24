"""
changes: kept Milestone 1 strict metrics, added 'relaxed' validity, also uniqueness/novelty over the relaxed molecules (as in authors' code).
valence histograms counting in .5 units, for aromatic bonds (to keep them stay integers)
"""
from collections import Counter

import torch
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from digress.data.chem import (build_mol, build_mol_with_partial_charges, flat_smiles, graph_to_mol,
                               largest_fragment)
from digress.data.info import QM9


def _sanitized(mol):
    try:
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def _flat(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return smiles if mol is None else flat_smiles(mol)


def relaxed_smiles(atoms, adj, mask, info=QM9):
    """DiGress author code: evaluation SMILES with relaxed construction (S+/N+/O+ allowed), 
    largest fragment only for disconnected molecules/ graphs; None if invalid
    """
    mol = _sanitized(build_mol_with_partial_charges(atoms, adj, mask, info))
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(largest_fragment(mol))
    except Exception:
        return None


def evaluate(atoms, bonds, masks, train_smiles_flat, info=QM9):
    """MS1 (strict) metrics and the relaxed ones:

    Strict: validity = RDKit sanitizes the neutral molecule; uniqueness / novelty over valid and(!) connected ones.
    DiGress author code version: relaxed_validity allows +1 charges on S/N/O and keeps the largest fragment; uniqueness_relaxed /
    novelty_relaxed computed over those SMILES as in authors' code.
    """
    n = len(atoms)
    fails, n_valid, n_connected, good, relaxed = Counter(), 0, 0, [], []
    for a, b, m in zip(atoms, bonds, masks):
        mol = build_mol(a, b, m, info)
        connected = len(Chem.GetMolFrags(mol)) == 1
        n_connected += connected
        try:
            Chem.SanitizeMol(mol)
        except Exception as err:
            fails["valence" if "valence" in str(err).lower() else "other"] += 1
        else:
            n_valid += 1
            if connected:
                good.append(flat_smiles(mol))
        s = relaxed_smiles(a, b, m, info)
        if s is not None:
            relaxed.append(s)
    unique = set(good)
    n_novel = sum(s not in train_smiles_flat for s in unique)
    unique_relaxed = set(relaxed)
    novel_relaxed = sum(_flat(s) not in train_smiles_flat for s in unique_relaxed)
    return {
        "n": n,
        "validity": n_valid / n,
        "connected": n_connected / n,
        "valid_connected": len(good) / n,
        "uniqueness": len(unique) / len(good) if good else 0.0,
        "novelty": n_novel / len(unique) if unique else 0.0,
        "vun": n_novel / n,
        "relaxed_validity": len(relaxed) / n,
        "uniqueness_relaxed": len(unique_relaxed) / len(relaxed) if relaxed else 0.0,
        "novelty_relaxed": novel_relaxed / len(unique_relaxed) if unique_relaxed else 0.0,
        "fail_valence": fails["valence"] / n,
        "fail_other": fails["other"] / n,
        "bonds_per_mol": (bonds.triu(1) > 0).sum((1, 2)).float().mean().item(),
    }


def _tv(p, q):
    return (
        0.5 * (p / p.sum().clamp(min=1) - q / q.sum().clamp(min=1)).abs().sum().item()
    )


def _valences(adj, mask, info):
    """The per-atom sum of bond orders in half units when aromatic bonds (order 1.5) exist so they stay integers"""
    scale = 2 if info.aromatic else 1
    orders = torch.tensor([round(scale * o) for o in info.bond_orders])
    pairs = mask[:, :, None] & mask[:, None, :]
    return (orders[adj] * pairs).sum(-1), scale


def graph_histograms(atoms, adj, mask, info=QM9):
    atoms, adj, mask = atoms.cpu(), adj.cpu().long(), mask.cpu().bool()
    max_atoms = atoms.shape[1]
    valence, scale = _valences(adj, mask, info)
    n_valence = scale * 3 * (max_atoms - 1) + 1
    pairs = mask[:, :, None] & mask[:, None, :]
    return {
        "n_atoms": torch.bincount(mask.sum(1), minlength=max_atoms + 1).float(),
        "atom_types": torch.bincount(atoms[mask], minlength=info.num_atom_types).float(),
        "bond_types": torch.bincount(adj[torch.triu(pairs, diagonal=1)], minlength=info.num_bond_types).float(),
        "valence": torch.bincount(
            (atoms * n_valence + valence)[mask], minlength=info.num_atom_types * n_valence
        ).float(),
    }


def distribution_distances(generated, reference):
    return {f"tv_{k}": _tv(generated[k], reference[k]) for k in reference}


def failure_breakdown(atoms, adj, mask, info=QM9):
    atoms, adj, mask = atoms.cpu(), adj.cpu().long(), mask.cpu().bool()
    valence, scale = _valences(adj, mask, info)
    over = (valence > scale * torch.tensor(info.max_valence)[atoms]) & mask
    valid = torch.tensor([graph_to_mol(a, b, m, info) is not None for a, b, m in zip(atoms, adj, mask)])
    return {
        "violation_rate": {el: over[mask & (atoms == k)].float().mean().item() for k, el in enumerate(info.atom_list)},
        "valid": valid.tolist(),
    }
