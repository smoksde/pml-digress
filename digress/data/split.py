from dataclasses import dataclass
from typing import NamedTuple

import torch
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from digress.config import CACHE_PATH, MAX_ATOMS, QM9_TARGETS
from digress.data.stats import property_values
from digress.data.chem import flat_smiles
from digress.data.dataset import get_dataset, load_cache
from digress.evaluation.similarity import FP_BITS, FP_RADIUS, nearest_similarity

SHIFT_TARGETS = ("mu", "alpha", "homo", "lumo", "gap")
BASELINE_SIZE = 1000


class PropertyShift(NamedTuple):
    name: str
    full_mean: float
    full_std: float
    subset_mean: float
    subset_std: float
    shift: float


@dataclass
class SplitCheck:
    size: int
    n_train: int
    n_val: int
    n_test: int
    leaked_val_graphs: int
    leaked_test_graphs: int
    atom_count_distance: float
    property_shifts: list
    test_similarity: torch.Tensor
    baseline_similarity: torch.Tensor


def _atom_count_distance(scan, data):
    full = torch.tensor(
        [n for n, ok in zip(scan.n_atoms, scan.usable.tolist()) if ok]
    ).float()
    p_full = torch.histc(full, MAX_ATOMS, 1, MAX_ATOMS + 1) / len(full)
    p_sub = torch.histc(data.masks.sum(1).float(), MAX_ATOMS, 1, MAX_ATOMS + 1) / len(
        data
    )
    return 0.5 * (p_full - p_sub).abs().sum().item()


def _property_shifts(scan, subset_y):
    rows = []
    for name in SHIFT_TARGETS:
        full, sub = property_values(scan, name), subset_y[:, QM9_TARGETS[name]]
        rows.append(
            PropertyShift(
                name,
                full.mean().item(),
                full.std().item(),
                sub.mean().item(),
                sub.std().item(),
                ((sub.mean() - full.mean()) / full.std()).item(),
            )
        )
    return rows


def check_split(scan):
    data = get_dataset()
    train = [
        flat_smiles(Chem.MolFromSmiles(data.smiles[i])) for i in data.train_idx.tolist()
    ]
    val = [
        flat_smiles(Chem.MolFromSmiles(data.smiles[i])) for i in data.val_idx.tolist()
    ]
    test = [
        flat_smiles(Chem.MolFromSmiles(data.smiles[i])) for i in data.test_idx.tolist()
    ]
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)
    train_fps = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in train]
    test_fps = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in test]
    train_set = set(train)
    return SplitCheck(
        size=len(data),
        n_train=len(train),
        n_val=len(val),
        n_test=len(test),
        leaked_val_graphs=sum(s in train_set for s in val),
        leaked_test_graphs=sum(s in train_set for s in test),
        atom_count_distance=_atom_count_distance(scan, data),
        property_shifts=_property_shifts(
            scan, load_cache(CACHE_PATH, need_labels=True)["y"]
        ),
        test_similarity=nearest_similarity(test_fps, train_fps),
        baseline_similarity=nearest_similarity(
            train_fps[:BASELINE_SIZE], train_fps, exclude_self=True
        ),
    )
