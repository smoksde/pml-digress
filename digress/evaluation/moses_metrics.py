"""
new file: moses_metrics.py for MOSES dataset; dist. metrics
SNN, Frag and Scaf as in the MOSES paper/ the authors' reference code

Morgan fingerprints with 1024 bits, BRICS fragments and scaffolds with >= 2 rings
FCD and Filters only computed if fcd_torch / molsets packages are available 
(local, but not yet on cluster .sif file)
"""

import warnings
from collections import Counter

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import BRICS, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

SNN_RADIUS, SNN_BITS = 2, 1024


def _mols(smiles):
    mols = [Chem.MolFromSmiles(s) for s in smiles if s]
    return [m for m in mols if m is not None]


def snn(gen_mols, ref_mols):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=SNN_RADIUS, fpSize=SNN_BITS)
    ref_fps = [gen.GetFingerprint(m) for m in ref_mols]
    return float(np.mean([max(DataStructs.BulkTanimotoSimilarity(gen.GetFingerprint(m), ref_fps))
                          for m in gen_mols]))


def _fragments(mol):
    # as in moses authors' code fragmenter: break BRICS bonds
    return Chem.MolToSmiles(BRICS.BreakBRICSBonds(mol)).split(".")


def _scaffold(mol, min_rings=2):
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    except Exception:
        return None
    if scaffold.GetRingInfo().NumRings() < min_rings:
        return None
    smiles = Chem.MolToSmiles(scaffold)
    return smiles or None


def _counts(mols, fn):
    counts = Counter()
    for m in mols:
        out = fn(m)
        if out is None:
            continue
        counts.update(out if isinstance(out, (set, list)) else [out])
    return counts


def cos_similarity(ref_counts, gen_counts):
    keys = sorted(set(ref_counts) | set(gen_counts))
    if not keys:
        return float("nan")
    r = np.array([ref_counts.get(k, 0) for k in keys], dtype=float)
    g = np.array([gen_counts.get(k, 0) for k in keys], dtype=float)
    denom = np.linalg.norm(r) * np.linalg.norm(g)
    return float(r @ g / denom) if denom else float("nan")


def _fcd(gen_smiles, ref_smiles):
    try:
        import torch
        from fcd_torch import FCD
    except ImportError:
        warnings.warn("fcd_torch not installed: FCD skipped") # can later remove once pml.sif has pkg
        return None
    return float(FCD(device="cuda" if torch.cuda.is_available() else "cpu", n_jobs=1)(gen_smiles, ref_smiles))


def _filters(gen_mols):
    try:
        from moses.metrics.metrics import fraction_passes_filters
    except ImportError:
        warnings.warn("molsets (moses) not installed: Filters skipped") # same
        return None
    return float(fraction_passes_filters(gen_mols))


def moses_metrics(gen_smiles, ref_smiles):
    """gen_smiles: valid generated SMILES (e.g. evaluation.metrics.relaxed_smiles); ref_smiles: test molecules."""
    gen_mols, ref_mols = _mols(gen_smiles), _mols(ref_smiles)
    if not gen_mols:
        return {"snn": None, "frag": None, "scaf": None, "fcd": None, "filters": None}
    return {
        "snn": snn(gen_mols, ref_mols),
        "frag": cos_similarity(_counts(ref_mols, _fragments), _counts(gen_mols, _fragments)),
        "scaf": cos_similarity(_counts(ref_mols, _scaffold), _counts(gen_mols, _scaffold)),
        "fcd": _fcd([Chem.MolToSmiles(m) for m in gen_mols], [Chem.MolToSmiles(m) for m in ref_mols]),
        "filters": _filters(gen_mols),
    }
