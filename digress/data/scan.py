from dataclasses import dataclass

import torch
from rdkit import Chem, RDLogger

from digress.config import MAX_ATOMS
from digress.data.chem import flat_smiles
from digress.data.dataset import load_qm9, molecule_to_record

BOND_NAMES = ("single", "double", "triple")


@dataclass
class QM9Scan:

    reasons: list
    symbols: list
    bond_counts: torch.Tensor
    n_rings: list
    smiles: list
    smiles_flat: list
    y: torch.Tensor

    @property
    def total(self):
        return len(self.reasons)

    @property
    def usable(self):
        return torch.tensor([r == "ok" for r in self.reasons])

    @property
    def n_atoms(self):
        return [len(s) for s in self.symbols]


def scan_qm9(max_atoms=MAX_ATOMS):
    RDLogger.DisableLog("rdApp.*")
    all_smiles, all_y = load_qm9()
    cols = {k: [] for k in ("reasons", "symbols", "bond_counts", "n_rings", "smiles", "smiles_flat")}
    for s in all_smiles:
        mol = Chem.MolFromSmiles(s)
        record, reason = molecule_to_record(mol, max_atoms)
        ok = record is not None
        cols["reasons"].append(reason or "ok")
        cols["symbols"].append(tuple(a.GetSymbol() for a in mol.GetAtoms()) if mol is not None else ())
        cols["bond_counts"].append(torch.bincount(record[2].triu(1).flatten(), minlength=4)[1:].tolist() if ok else [0] * 3)
        cols["n_rings"].append(mol.GetRingInfo().NumRings() if ok else -1)
        cols["smiles"].append(Chem.MolToSmiles(mol) if ok else "")
        cols["smiles_flat"].append(flat_smiles(mol) if ok else "")
    cols["bond_counts"] = torch.tensor(cols["bond_counts"])
    return QM9Scan(**cols, y=all_y)
