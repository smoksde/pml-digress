"""
Static description of a molecular dataset: elements, bond classes and padding size.

Stuff that was const. for whole module for only-QM9 setting (atom list, valences, max atoms, ..) 
now here, can rest of code can now works with other datasets by passing `DatasetInfo`
"""
from dataclasses import dataclass, replace

from rdkit import Chem

_KEKULE_BONDS = (Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE)
_KEKULE_ORDERS = (0.0, 1.0, 2.0, 3.0)


@dataclass(frozen=True)
class DatasetInfo:
    name: str
    atom_list: tuple
    max_valence: tuple
    default_valence: tuple
    atom_weights: tuple
    max_atoms: int
    max_weight: float
    aromatic: bool = False  # True: aromatic bonds are a separate edge class/type

    @property
    def key(self):
        return self.name + ("_arom" if self.aromatic else "")

    @property
    def num_atom_types(self):
        return len(self.atom_list)

    @property
    def bond_types(self):
        """RDKit bond type of edge class k + 1 (class 0 = no bond)."""
        return _KEKULE_BONDS + ((Chem.BondType.AROMATIC,) if self.aromatic else ())

    @property
    def bond_orders(self):
        """Bond order of each edge class, including class 0 (no bond)."""
        return _KEKULE_ORDERS + ((1.5,) if self.aromatic else ())

    @property
    def bond_labels(self):
        return ("none", "single", "double", "triple") + (("aromatic",) if self.aromatic else ())

    @property
    def num_bond_types(self):
        return len(self.bond_orders)

    @property
    def num_edges(self):
        return self.max_atoms * (self.max_atoms - 1) // 2


QM9 = DatasetInfo(
    name="qm9",
    atom_list=("C", "N", "O", "F"),
    max_valence=(4, 3, 2, 1),
    default_valence=(4, 3, 2, 1),
    atom_weights=(12.0, 14.0, 16.0, 19.0),
    max_atoms=9,
    max_weight=150.0,
)

MOSES = DatasetInfo(
    name="moses",
    atom_list=("C", "N", "S", "O", "F", "Cl", "Br"),
    max_valence=(4, 3, 6, 2, 1, 1, 1),
    default_valence=(4, 3, 4, 2, 1, 1, 1),  #S atom: DiGress author code using 4
    atom_weights=(12.0, 14.0, 32.0, 16.0, 19.0, 35.4, 79.9),
    max_atoms=27,
    max_weight=350.0,
)

DATASETS = {"qm9": QM9, "moses": MOSES}


def get_info(name, aromatic=False):
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; choose from {sorted(DATASETS)}")
    return replace(DATASETS[name], aromatic=bool(aromatic))
