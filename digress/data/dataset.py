"""
Some changes allowing for new MOSES dataset:


get_dataset() now takes a dataset `name` so more than just QM9 data

QM9: keep using same splits ratio but: using all data, for subset of 50000

use get_dataset(subset_size=50000, ...)

MOSES: already provided splits by the authors/ github repo for the paper.

For MOSES cache/ u8tin tensors rather than long (might change to long..)
also, store the split indicess once.

For MOSES, --subset_size gives a view rather than generating a specific cache

subsample() method for an order-preserving, seeded sampling whenever we would have to 
iterate over all molecules for eval
"""

import random
from collections import Counter
from dataclasses import dataclass, field
from functools import cached_property, lru_cache
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem, RDLogger

from digress.config import (
    CACHE_DIR,
    DATA_DIR,
    DATA_WORKERS,
    QM9_TARGETS,
    RANDOM_SEED,
    TRAIN_FRACTION,
    VAL_FRACTION,
)
from digress.data.chem import flat_smiles, mol_to_arrays
from digress.data.graphs import edge_vector_to_adj
from digress.data.info import QM9, get_info

CACHE_VERSION = 3   # set ver. 3 here for this branch
SKIP_REASONS = (
    "unparsable",
    "too_many_atoms",
    "unsupported_element",
    "charged",
    "conversion_failed",
)


def molecule_to_record(mol, info=QM9):
    if mol is None:
        return None, "unparsable"
    if mol.GetNumAtoms() > info.max_atoms:
        return None, "too_many_atoms"
    if any(a.GetSymbol() not in info.atom_list for a in mol.GetAtoms()):
        return None, "unsupported_element"
    if any(a.GetFormalCharge() != 0 for a in mol.GetAtoms()):
        return None, "charged"
    try:
        return mol_to_arrays(mol, info), None
    except Exception:
        return None, "conversion_failed"


def _process_smiles(args):
    smiles, info = args
    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    record, reason = molecule_to_record(mol, info)
    if record is None:
        return None, reason, None, None
    return record, None, Chem.MolToSmiles(mol), flat_smiles(mol)


def _process_all(smiles, info, workers):
    jobs = ((s, info) for s in smiles)
    if workers <= 1:
        yield from map(_process_smiles, jobs)
        return
    with Pool(workers) as pool:
        yield from pool.imap(_process_smiles, jobs, chunksize=2000)


class _Collector:
    def __init__(self):
        self.atoms, self.counts, self.edges, self.smiles, self.flat, self.keep = [], [], [], [], [], []
        self.skipped, self.duplicates = Counter(), 0

    def add(self, result, source_index, seen=None):
        record, reason, smiles, flat = result
        if record is None:
            self.skipped[reason] += 1
            return False
        if seen is not None:
            if flat in seen:
                self.duplicates += 1
                return False
            seen.add(flat)
        atoms, n, edges = record
        self.atoms.append(atoms)
        self.counts.append(n)
        self.edges.append(edges)
        self.smiles.append(smiles)
        self.flat.append(flat)
        self.keep.append(source_index)
        return True

    def tensors(self, max_atoms):
        counts = torch.tensor(self.counts)
        return (
            torch.from_numpy(np.stack(self.atoms)),
            torch.arange(max_atoms)[None, :] < counts[:, None],
            torch.from_numpy(np.stack(self.edges)),
        )


def load_qm9():
    from torch_geometric.datasets import QM9 as PygQM9

    ds = PygQM9(root=DATA_DIR)
    return ds._data.smiles, ds._data.y


def _split_indices(
        n, train_fraction=TRAIN_FRACTION, val_fraction=VAL_FRACTION, seed=RANDOM_SEED
):
    perm = list(range(n))
    random.Random(seed).shuffle(perm)
    n_train = int(n * train_fraction)
    n_val = int(n_train * val_fraction)
    train, test = perm[:n_train], perm[n_train:]
    return torch.tensor(train[n_val:]), torch.tensor(train[:n_val]), torch.tensor(test)


def _build_qm9(info, subset_size, workers):
    """Same as MS1: shuffle w/ seed, filter, drop dupes, rnd split but by default over whole dataset; 
    Use --subset_size to override and get the M1 data for value 50000.
    """
    all_smiles, all_y = load_qm9()
    order = list(range(len(all_smiles)))
    random.Random(RANDOM_SEED).shuffle(order)
    col, seen = _Collector(), set()
    for i, result in zip(order, _process_all((all_smiles[i] for i in order), info, workers)):
        col.add(result, i, seen)
        if subset_size and len(col.atoms) >= subset_size:
            break
    atoms, masks, edges = col.tensors(info.max_atoms)
    train_idx, val_idx, test_idx = _split_indices(len(atoms))
    return {
        "atoms": atoms, "masks": masks, "edge_vecs": edges, "smiles": col.smiles,
        "train_flat": [col.flat[i] for i in train_idx.tolist()],
        "split": {"train": train_idx, "val": val_idx, "test": test_idx},
        "y": all_y[torch.tensor(col.keep)], "skipped": dict(col.skipped), "skipped_duplicates": col.duplicates,
    }


def _build_moses(info, workers):
    from digress.data.moses import load_moses_smiles

    col, parts, start = _Collector(), {}, 0
    for split, part in (("train", "train"), ("test_scaffolds", "val"), ("test", "test")):
        seen = set() if part == "train" else None  # just as a sanity check; MOSES data should be clean (de-duped) though
        for i, result in enumerate(_process_all(load_moses_smiles(split), info, workers)):
            col.add(result, i, seen)
        parts[part] = torch.arange(start, len(col.atoms))
        start = len(col.atoms)
        print(f"  moses/{split}: {len(parts[part])} molecules", flush=True)
    atoms, masks, edges = col.tensors(info.max_atoms)
    return {
        "atoms": atoms, "masks": masks, "edge_vecs": edges, "smiles": col.smiles,
        "train_flat": col.flat[: len(parts["train"])], "split": parts, "y": None,
        "skipped": dict(col.skipped), "skipped_duplicates": col.duplicates,
    }


def cache_path(name, subset_size=None, aromatic=False):
    sub = f"_sub{subset_size}" if (subset_size and name == "qm9") else ""  # MOSES --subset_size gives view of full /cache file
    return CACHE_DIR / f"{name}{sub}{'_arom' if aromatic else ''}_cache.pt"


def load_cache(name="qm9", subset_size=None, aromatic=False, workers=DATA_WORKERS):
    path = cache_path(name, subset_size, aromatic)
    if Path(path).exists():
        cache = torch.load(path, weights_only=False)
        if cache.get("version") == CACHE_VERSION:
            return cache
    info = get_info(name, aromatic)
    print(f"building dataset cache {path} ...", flush=True)
    cache = _build_qm9(info, subset_size, workers) if name == "qm9" else _build_moses(info, workers)
    cache.update(version=CACHE_VERSION, name=name, aromatic=bool(aromatic))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, path)
    print(f"cached {len(cache['atoms'])} molecules, skipped {sum(cache['skipped'].values())} (filter: "
          f"{cache['skipped']}) + {cache['skipped_duplicates']} (duplicate graphs)", flush=True)
    return cache


def _label_column(y, label_property):
    if label_property not in QM9_TARGETS:
        raise ValueError(f"Unknown label property {label_property!r}; choose from {sorted(QM9_TARGETS)}")
    return y[:, QM9_TARGETS[label_property]].unsqueeze(-1).float()


def subsample(idx, cap, seed=RANDOM_SEED):
    """Order-preserving, seeded subset of maximum `cap` indices; all of them if cap is None"""
    if cap is None or len(idx) <= cap:
        return idx
    gen = torch.Generator().manual_seed(seed)
    return idx[torch.randperm(len(idx), generator=gen)[:cap].sort().values]


@dataclass
class GraphDataset:
    info: object
    atoms: torch.Tensor  # (N, max_atoms)
    masks: torch.Tensor  # (N, max_atoms)
    edge_vecs: torch.Tensor  # (N, num_edges), upper triangle row-major ; atoms, masks, edge_vecs of type uint8, bool, uint8, respectively.
    smiles: list
    train_idx: torch.Tensor
    val_idx: torch.Tensor
    test_idx: torch.Tensor
    train_flat: list = field(repr=False, default_factory=list)
    meta: dict = field(repr=False, default_factory=dict)
    y: torch.Tensor = None
    y_mean: torch.Tensor = None
    y_std: torch.Tensor = None

    def __len__(self):
        return len(self.atoms)

    @property
    def name(self):
        return self.info.name

    @cached_property
    def train_smiles_flat(self):
        return set(self.train_flat)

    def batch(self, idx):
        return (
            self.atoms[idx].long(),
            self.masks[idx],
            self.edge_vecs[idx].long(),
            (self.y[idx] if self.y is not None else None),
        )

    def graphs(self, idx):
        return (self.atoms[idx].long(), edge_vector_to_adj(self.edge_vecs[idx].long(), self.info.max_atoms),
                self.masks[idx])

    def sample_batch(self, batch_size):
        return self.batch(self.train_idx[torch.randint(0, len(self.train_idx), (batch_size,))])


@lru_cache(maxsize=None)
def get_dataset(name="qm9", subset_size=None, aromatic=False, label_property=None):
    """--subset_size: QM9 -> first N usable molecules of shuffle (seeded) like in MS1;
                      MOSES -> seeded subset of official train split (val/test split stay complete; evaluation caps them separately)
    """
    info = get_info(name, aromatic)
    cache = load_cache(name, subset_size, aromatic)
    split = cache["split"]
    train_idx = split["train"]
    if name == "moses" and subset_size:
        train_idx = subsample(train_idx, subset_size)
    train_flat = cache["train_flat"]
    if len(train_idx) != len(split["train"]):
        position = {int(k): p for p, k in enumerate(split["train"].tolist())}
        train_flat = [train_flat[position[k]] for k in train_idx.tolist()]
    data = GraphDataset(
        info, cache["atoms"], cache["masks"], cache["edge_vecs"], cache["smiles"], train_idx, split["val"],
        split["test"], train_flat,
        meta={"skipped": cache["skipped"], "skipped_duplicates": cache["skipped_duplicates"], "y_all": cache["y"],
              "subset_size": subset_size},
    )
    if label_property is not None:
        if cache["y"] is None:
            raise ValueError(f"dataset {name!r} has no labels; unset LABEL_PROPERTY")
        data.y = _label_column(cache["y"], label_property)
        data.y_mean, data.y_std = data.y[train_idx].mean(0), data.y[train_idx].std(0)
    return data
