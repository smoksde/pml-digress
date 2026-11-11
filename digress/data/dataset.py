import random
from dataclasses import dataclass
from functools import cached_property, lru_cache
from pathlib import Path

import torch
from rdkit import Chem

from digress.config import (CACHE_PATH, DATA_DIR, LABEL_PROPERTY, MAX_ATOMS, QM9_SUBSET_SIZE, QM9_TARGETS, RANDOM_SEED,
                            TRAIN_FRACTION)
from digress.data.chem import ATOM_LIST, flat_smiles, mol_to_graph
from digress.data.graphs import adj_to_edge_vector

SKIP_REASONS = ("unparsable", "too_many_atoms", "unsupported_element", "charged", "conversion_failed")


def molecule_to_record(mol, max_atoms=MAX_ATOMS):
    if mol is None:
        return None, "unparsable"
    if mol.GetNumAtoms() > max_atoms:
        return None, "too_many_atoms"
    if any(a.GetSymbol() not in ATOM_LIST for a in mol.GetAtoms()):
        return None, "unsupported_element"
    if any(a.GetFormalCharge() != 0 for a in mol.GetAtoms()):
        return None, "charged"
    try:
        return mol_to_graph(mol, max_atoms), None
    except Exception:
        return None, "conversion_failed"


def load_qm9():
    from torch_geometric.datasets import QM9

    ds = QM9(root=DATA_DIR)
    return ds._data.smiles, ds._data.y


def _build_cache(path, subset_size=QM9_SUBSET_SIZE, max_atoms=MAX_ATOMS):
    all_smiles, all_y = load_qm9()
    order = list(range(len(all_smiles)))
    random.Random(RANDOM_SEED).shuffle(order)

    records, smiles, ys, skipped = [], [], [], 0
    for i in order:
        if len(records) >= subset_size:
            break
        mol = Chem.MolFromSmiles(all_smiles[i])
        record, _ = molecule_to_record(mol, max_atoms)
        if record is None:
            skipped += 1
            continue
        records.append(record)
        smiles.append(Chem.MolToSmiles(mol))
        ys.append(all_y[i])

    atoms, masks, bonds = (torch.stack(col) for col in zip(*records))
    cache = {"atoms": atoms, "masks": masks, "bonds": bonds, "smiles": smiles, "y": torch.stack(ys)}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, path)
    print(f"cached {len(records)} molecules, skipped {skipped}")
    return cache


def load_cache(path=CACHE_PATH, need_labels=False):
    if Path(path).exists():
        cache = torch.load(path)
        if not need_labels or "y" in cache:
            return cache
    return _build_cache(path)


def _split_indices(n, train_fraction=TRAIN_FRACTION, seed=RANDOM_SEED):
    perm = list(range(n))
    random.Random(seed).shuffle(perm)
    n_train = int(n * train_fraction)
    return torch.tensor(perm[:n_train]), torch.tensor(perm[n_train:])


def _label_column(cache, label_property):
    if label_property not in QM9_TARGETS:
        raise ValueError(f"Unknown label property {label_property!r}; choose from {sorted(QM9_TARGETS)}")
    return cache["y"][:, QM9_TARGETS[label_property]].unsqueeze(-1).float()


@dataclass
class GraphDataset:
    atoms: torch.Tensor
    masks: torch.Tensor
    edge_vecs: torch.Tensor
    smiles: list
    train_idx: torch.Tensor
    test_idx: torch.Tensor
    y: torch.Tensor = None
    y_mean: torch.Tensor = None
    y_std: torch.Tensor = None

    def __len__(self):
        return len(self.atoms)

    @cached_property
    def train_smiles_flat(self):
        smiles = (self.smiles[i] for i in self.train_idx.tolist())
        return {flat_smiles(Chem.MolFromSmiles(s)) for s in smiles}

    def batch(self, idx):
        return self.atoms[idx], self.masks[idx], self.edge_vecs[idx], (self.y[idx] if self.y is not None else None)

    def sample_batch(self, batch_size):
        return self.batch(self.train_idx[torch.randint(0, len(self.train_idx), (batch_size,))])


@lru_cache(maxsize=None)
def get_dataset(label_property=LABEL_PROPERTY, cache_path=CACHE_PATH):
    cache = load_cache(cache_path, need_labels=label_property is not None)
    train_idx, test_idx = _split_indices(len(cache["atoms"]))
    data = GraphDataset(cache["atoms"], cache["masks"], adj_to_edge_vector(cache["bonds"]), cache["smiles"],
                        train_idx, test_idx)
    if label_property is not None:
        data.y = _label_column(cache, label_property)
        data.y_mean, data.y_std = data.y[train_idx].mean(0), data.y[train_idx].std(0)
    return data
