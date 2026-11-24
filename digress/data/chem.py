"""
Changes to make this more consistent with paper/ authors' code, most notably: 

using the "relaxed" construction with partial charges, S N O exceeding neutral valence

by 1 get a +1 formal charge. the two atoms of a bond are checked directly, rather than
pardsing the error msg of rdkit.
"""

import numpy as np
import torch
from rdkit import Chem

from digress.data.info import QM9

#QM9 names, kept for MS1 dataset report (scan/stats/data_quality). New code using a DatasetInfo
ATOM_LIST = list(QM9.atom_list)
MAX_VALENCE = list(QM9.max_valence)
NUM_ATOM_TYPES = QM9.num_atom_types
BOND_ORDER = list(QM9.bond_types)
NUM_BOND_TYPES = QM9.num_bond_types

# valence over which "relaxed" scheme adds +1 chjarge
_CHARGEABLE = {"N": 3, "O": 2, "S": 2}


def edge_position(i, j, n):
    """Idx of atom pair (i, j), i < j, in row-major upper-triangle edge vector of the graph on n atoms"""
    return i * n - i * (i + 1) // 2 + (j - i - 1)


def mol_to_arrays(mol, info=QM9):
    """The compact graph encoding used for /cache: atom classes, atom count and edge-class vector"""
    mol = Chem.Mol(mol)
    if not info.aromatic:
        Chem.Kekulize(mol, clearAromaticFlags=True)
    n_max = info.max_atoms
    atoms = np.zeros(n_max, dtype=np.uint8)
    for i, atom in enumerate(mol.GetAtoms()):
        atoms[i] = info.atom_list.index(atom.GetSymbol())
    edges = np.zeros(info.num_edges, dtype=np.uint8)
    bond_types = info.bond_types
    for bond in mol.GetBonds():
        i, j = sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        edges[edge_position(i, j, n_max)] = bond_types.index(bond.GetBondType()) + 1
    return atoms, mol.GetNumAtoms(), edges


def mol_to_graph(mol, info=QM9):
    """Dense encoding (atom classes, node mask, symmetric bond-class matrix) as torch tensors"""
    atoms, n, edges = mol_to_arrays(mol, info)
    mask = torch.arange(info.max_atoms) < n
    i, j = torch.triu_indices(info.max_atoms, info.max_atoms, offset=1)
    adj = torch.zeros(info.max_atoms, info.max_atoms, dtype=torch.long)
    adj[i, j] = torch.from_numpy(edges).long()
    return torch.from_numpy(atoms).long(), mask, adj + adj.T


def _bonds(bond_adj, mask):
    keep = torch.triu(bond_adj, diagonal=1) * (mask[:, None] & mask[None, :])
    return [(i, j, int(keep[i, j])) for i, j in torch.nonzero(keep).tolist()]


def build_mol(atom_idx, bond_adj, mask, info=QM9):
    mol = Chem.RWMol()
    mask = mask.bool()
    idx_map = {}
    for i, a in enumerate(atom_idx.tolist()):
        if mask[i]:
            idx_map[i] = mol.AddAtom(Chem.Atom(info.atom_list[a]))
    for i, j, b in _bonds(bond_adj, mask):
        mol.AddBond(idx_map[i], idx_map[j], info.bond_types[b - 1])
    return mol.GetMol()


def build_mol_with_partial_charges(atom_idx, bond_adj, mask, info=QM9):
    """DiGress' "relaxed" construction: S N or O whose valence exceeds the neutral one by 1 gets a +1
    formal charge (cf. and compare authors' code at src/analysis/rdkit_functions.py). 
    Checking the two atoms of each new bond
    directly rather than parsing error message"""
    mol = Chem.RWMol()
    mask = mask.bool()
    idx_map = {}
    for i, a in enumerate(atom_idx.tolist()):
        if mask[i]:
            idx_map[i] = mol.AddAtom(Chem.Atom(info.atom_list[a]))
    for i, j, b in _bonds(bond_adj, mask):
        mol.AddBond(idx_map[i], idx_map[j], info.bond_types[b - 1])
        for k in (idx_map[i], idx_map[j]):
            atom = mol.GetAtomWithIdx(k)
            if atom.GetFormalCharge() != 0 or atom.GetSymbol() not in _CHARGEABLE:
                continue
            try:
                atom.UpdatePropertyCache(strict=True)
            except Exception:
                valence = round(sum(bd.GetBondTypeAsDouble() for bd in atom.GetBonds()))
                if valence - _CHARGEABLE[atom.GetSymbol()] == 1:
                    atom.SetFormalCharge(1)
    return mol.GetMol()


def graph_to_mol(atom_idx, bond_adj, mask, info=QM9):
    mol = build_mol(atom_idx, bond_adj, mask, info)
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return mol


def flat_smiles(mol):
    return Chem.MolToSmiles(mol, isomericSmiles=False)


def largest_fragment(mol):
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    return max(frags, default=mol, key=lambda m: m.GetNumAtoms())
